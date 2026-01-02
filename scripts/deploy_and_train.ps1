# TPU Deployment & Training Script
$ips = @("34.74.0.31", "34.26.29.55", "34.74.128.78", "34.148.46.138")
$key = "C:\Users\mesof\.ssh\google_compute_engine"
$user = "mesof"

# 1. Deploy Code to ALL workers (Using Git for speed)
# We clone/pull the repo instead of SCP-ing everything
foreach ($ip in $ips) {
    Write-Host "Deploying code to $ip via Git..."
    $deploy_cmd = "if [ ! -d 'cift-markets' ]; then git clone https://github.com/MessoJ/cift-markets.git; else cd cift-markets && git pull; fi"
    ssh -i $key -o StrictHostKeyChecking=no $user@$ip $deploy_cmd
}

# 2. Install Dependencies on ALL workers (Parallel would be better, but sequential is safer for now)
# Note: We need to install dependencies. We also ensure we are in the right directory if there's a requirements file, 
# but here we install specific packages manually as before.
$install_cmd = "sudo apt-get update && sudo apt-get install -y libopenblas-base && sudo python3 -m pip install 'numpy<2' pandas pyarrow gcsfs torch~=2.2.0 torch_xla[tpu]~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html"

foreach ($ip in $ips) {
    Write-Host "Installing dependencies on $ip..."
    ssh -i $key -o StrictHostKeyChecking=no $user@$ip $install_cmd
}

# 3. Run Training (Must be launched on all workers simultaneously)
# We use Start-Job to run them in parallel
$jobs = @()
foreach ($ip in $ips) {
    Write-Host "Starting training on $ip..."
    $job = Start-Job -ScriptBlock {
        param($ip, $key, $user)
        # Note the path change: scripts/gcp_tpu_train.py -> cift-markets/scripts/gcp_tpu_train.py
        ssh -i $key -o StrictHostKeyChecking=no $user@$ip "nohup sudo python3 cift-markets/scripts/gcp_tpu_train.py > tpu_train.log 2>&1 &"
    } -ArgumentList $ip, $key, $user
    $jobs += $job
}

# Wait for jobs
Write-Host "Training started. Waiting for output..."
Receive-Job -Job $jobs -Wait
