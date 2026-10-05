# AWS deployment

One EC2 instance runs the same Docker Compose stack as local development (PostgreSQL,
migrations, API), with the model baked into an image stored in ECR. It is meant to run
**on demand**: start it before a demo, stop it afterwards. Everything is created and
removed by Terraform.

```
 GitHub Actions ── OIDC (no keys) ──► IAM role ──► push image to ECR
       │                                     └──► record release in SSM Parameter Store
       └──► Session Manager command ──► EC2 instance (if running): pull release, restart, smoke test

 EC2 c7i-flex.large (Amazon Linux 2023, default VPC, port 80 only, no SSH)
   systemd ──► start.sh ──► docker compose: db (PostgreSQL 16) ─ migrate ─ api :80
   container logs ──► CloudWatch Logs          CPU quiet 2 h ──► CloudWatch alarm stops it
```

| Resource | Purpose |
| --- | --- |
| ECR repository | Private image registry; keeps the 3 newest images, scans on push |
| EC2 instance + 20 GB gp3 disk | Runs the stack; disk keeps the database while stopped |
| Security group | Inbound port 80 only (`allowed_cidrs`); no SSH port |
| Instance IAM role | Pull from this ECR repo, read its two SSM parameters, write logs, Session Manager |
| GitHub OIDC provider + deploy role | Lets workflows on `main` of this repo push images and roll out, nothing else |
| SSM parameters | `/review-classifier/image` (the release), `/review-classifier/postgres-password` (SecureString) |
| CloudWatch log group | Container logs, kept 14 days |
| CloudWatch alarm | Stops the instance after 2 hours below 5% CPU |
| AWS Budget | Emails at 80% and 100% of $10/month of usage (credits excluded) |

## One-time setup (about 30 minutes)

### 1. Tools on your Mac

```bash
brew install awscli
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
brew install --cask session-manager-plugin
```

`aws --version` should report 2.32 or newer.

### 2. Sign the command line in with your console user

`aws login` uses your normal console sign-in (with MFA) and gives the command line
short-lived credentials, so no access keys are created. If it reports that access is
denied, attach the AWS managed policy `SignInLocalDevelopmentAccess` to `mohammad-admin`
(IAM → Users → mohammad-admin → Add permissions) and try again.

```bash
aws login --profile admin
```

When asked for a region, enter `eu-north-1`. A browser window opens; sign in as
`mohammad-admin`. Terraform cannot read these credentials directly yet, so add a second
profile that hands them over, and use it from now on:

```bash
cat >> ~/.aws/config <<'EOF'

[profile rc]
credential_process = aws configure export-credentials --profile admin --format process
region = eu-north-1
EOF
export AWS_PROFILE=rc
aws sts get-caller-identity
```

The last command should print your account and `user/mohammad-admin`. The session lasts
up to 12 hours; after that, run `aws login --profile admin` again. Put
`export AWS_PROFILE=rc` in `~/.zshrc` to avoid typing it in every terminal.

### 3. Create the infrastructure

From the repository folder:

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
open -e terraform.tfvars
```

Set `alert_email` to your address and save. GitHub repositories created after 15 July
2026 (this one included) sign in to AWS with their numeric owner and repository IDs, so
add those too:

```bash
curl -s https://api.github.com/repos/mohammadabdalaziz241/review-classifier-service \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("github_owner_id =", d["owner"]["id"]); print("github_repository_id =", d["id"])' \
  | tee -a terraform.tfvars
```

Then:

```bash
terraform init
terraform plan
terraform apply
```

Read the plan (about 20 resources, nothing destroyed) and type `yes`. This creates the
instance **running**; its first boot installs Docker (about 2 minutes), then it waits for
a release. Left alone, it stops itself after two quiet hours. Commit the generated
`.terraform.lock.hcl` so CI uses the same provider versions. `terraform.tfstate` is
git-ignored: it contains the database password, so keep it private and back it up.

`apply` prints the values for the next step (`terraform output next_steps` repeats them).

### 4. Connect GitHub

In the repository on GitHub: **Settings → Secrets and variables → Actions**.

| Type | Name | Value |
| --- | --- | --- |
| Variable | `AWS_ROLE_ARN` | `github_deploy_role_arn` from `terraform output` |
| Variable | `AWS_REGION` | `eu-north-1` |
| Secret | `HF_TOKEN` | A Hugging Face **read** token (the models are private) |

### 5. First deploy

**Actions → Deploy → Run workflow**, choose `sentiment`. The workflow builds the image
with the pinned model, pushes it, records it as the release and, since the instance is
running, rolls it out and runs the smoke test against the live URL. The first build takes
about 10–15 minutes; later builds reuse cached layers. The run summary shows the URL.

Check it yourself, then stop it:

```bash
cd ..
scripts/aws.sh status
scripts/aws.sh stop
```

## Everyday use

```bash
scripts/aws.sh start
```

Starts the instance, waits until the API answers (about 2–3 minutes: boot, then loading
the model) and prints the URL, the `/docs` page and the model version. The address
changes on every start.

```bash
scripts/aws.sh stop
```

Stops it and prints how long it ran and roughly what that cost. The disk, the database
and the deployed release are kept.

| Command | What it does |
| --- | --- |
| `scripts/aws.sh status` | State, URL, time running and its cost so far, current release |
| `scripts/aws.sh logs` | Last 30 minutes of container logs; `scripts/aws.sh logs -f` to follow |
| `scripts/aws.sh shell` | Root shell on the instance through Session Manager |

**Forgot to stop it?** After 2 hours with CPU below 5% (5-minute averages), a CloudWatch
alarm stops it, so a forgotten instance costs about $0.20, not $70 a month. Occasional
demo requests barely move a 5-minute average, so treat 2 hours as the longest session:
for a longer one, run `scripts/aws.sh start` again after it stops, or raise
`idle_stop_minutes` in `terraform.tfvars` and run `terraform apply`.

**Switch model:** run the Deploy workflow again and choose `sarcasm`. If the instance is
stopped, it picks the new release up on its next start.

**Roll back:** releases are image digests; ECR keeps the last three. Set
`/review-classifier/image` back to an earlier digest (ECR console → repository → image →
digest) and run `scripts/aws.sh shell`, then `sudo /opt/review-classifier/start.sh`.

## Costs

Prices for eu-north-1 (Stockholm), on-demand, as of October 2026. Check the
[EC2 pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for current values.

**While running: about $0.096 per hour**

| Item | Price |
| --- | --- |
| c7i-flex.large (2 vCPU, 4 GiB) | $0.0908 / hour |
| Public IPv4 address | $0.005 / hour, only while running (released when stopped) |

**All the time, running or stopped: about $2 per month**

| Item | Price |
| --- | --- |
| 20 GB gp3 disk | $0.0836 / GB-month → $1.67 / month |
| ECR, about 1 GB per image × 3 kept | $0.10 / GB-month → about $0.30 / month |
| CloudWatch Logs | Kilobytes per day; within the always-free 5 GB / month |
| CloudWatch alarm (1) | Within the always-free 10 alarms |
| SSM parameters, Session Manager, IAM, OIDC | Free |
| AWS Budgets | Free for alert-only budgets |
| Data transfer | ECR → EC2 in the same region is free; API responses are tiny and the first 100 GB / month out are free |

GitHub Actions minutes are free for public repositories.

**Estimates for the credit period (until 3 April 2027, about 6 months)**

| Usage | Running hours | Running cost | Fixed cost | Total |
| --- | ---: | ---: | ---: | ---: |
| Occasional demos, 5 h / month | 30 | $2.90 | $12 | **about $15** |
| Regular, 5 h / week | 130 | $12.50 | $12 | **about $25** |
| Heavy, 2 h / day | 360 | $34.50 | $12 | **about $47** |
| Left running a whole month (no auto-stop) | 730 | $70 | — | $70 / month |

All of these fit in the $100 credit. The $10 monthly budget alert would only fire after
about 85 running hours in a month. Actual usage appears in **Billing → Free Tier** and
**Cost Explorer**, with up to a day's delay.

## Remove everything

```bash
cd infra
terraform destroy
```

This deletes the instance, its disk and database, the images and every other resource
above, including the budget. Do it before the credits or the Free plan period end (3 April
2027) if you no longer need the deployment.

## Security notes

- No SSH and no port 22: shell access and deploys go through Session Manager, authorised
  by IAM.
- No long-lived AWS keys anywhere: your Mac uses `aws login`, GitHub uses OIDC, and the
  instance uses its role. The GitHub role only trusts workflows on `main` of this
  repository and can only push to this ECR repository and run commands on this instance.
- IMDSv2 only, with a hop limit of 1, so containers cannot read the instance's credentials.
- The database password is generated by Terraform, stored as an SSM SecureString, and
  written on the instance to a root-only file. It is also in `terraform.tfstate`.
- The Hugging Face token is a GitHub secret, used only during the image build as a build
  secret; it is not in the image.
- [checkov](https://www.checkov.io/) passes all checks except nine, each explained by a
  `checkov:skip` comment next to the resource (for example, a public IP instead of a load
  balancer that would cost about $18 / month).

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Deploy: `Not authorized to perform sts:AssumeRoleWithWebIdentity` | `AWS_ROLE_ARN` is wrong; the workflow ran on a branch other than `main`; or `github_owner_id` / `github_repository_id` are missing or wrong (`terraform output github_oidc_subject` shows what the role accepts) |
| Deploy: `Missing repository settings` | A variable or secret from step 4 is not set |
| `scripts/aws.sh start` waits and times out | `scripts/aws.sh logs`; on the instance (`scripts/aws.sh shell`): `sudo journalctl -u review-classifier` and `sudo cat /var/log/cloud-init-output.log` |
| `aws: error: ... login` or expired credentials | Run `aws login --profile admin` again |
| `terraform apply` fails on the OIDC provider: already exists | Set `create_github_oidc_provider = false` in `terraform.tfvars` |
| An error mentioning the Free plan | That service or size is not available on the Free plan; tell me which resource failed |
