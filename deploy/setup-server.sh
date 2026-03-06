#!/usr/bin/env bash
# ===========================================================================
# AgriPay Compute Backend — Server Setup Script
# Target: Fresh Ubuntu 24.04 LTS with 256 GB SSD + 2 TB HDD
# ===========================================================================
set -euo pipefail

echo "========================================="
echo " AgriPay Server Setup"
echo "========================================="

# ------------------------------------------------------------------
# 1. System packages
# ------------------------------------------------------------------
echo "[1/7] Installing system packages..."
sudo apt-get update
sudo apt-get install -y \
    curl git nginx ufw \
    apt-transport-https ca-certificates software-properties-common

# ------------------------------------------------------------------
# 2. Mount 2 TB HDD at /mnt/data (skip if already mounted)
# ------------------------------------------------------------------
echo "[2/7] Setting up data drive..."
DATA_DIR="/mnt/data"
IMAGERY_DIR="$DATA_DIR/imagery"

if mountpoint -q "$DATA_DIR"; then
    echo "  $DATA_DIR is already mounted."
else
    echo "  Attempting to identify the 2 TB drive..."
    # List available drives for the user
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT | grep -E "disk|part"
    echo ""
    echo "  !! ACTION REQUIRED !!"
    echo "  If the HDD is not mounted yet, run these commands manually:"
    echo "    sudo mkfs.ext4 /dev/sdX    # replace sdX with the correct device"
    echo "    sudo mkdir -p $DATA_DIR"
    echo "    sudo mount /dev/sdX $DATA_DIR"
    echo "    echo '/dev/sdX $DATA_DIR ext4 defaults 0 2' | sudo tee -a /etc/fstab"
    echo ""
    read -p "  Press Enter once the drive is mounted (or Ctrl+C to abort)..."
fi

sudo mkdir -p "$IMAGERY_DIR"
sudo chown -R $USER:$USER "$DATA_DIR"
echo "  Imagery directory: $IMAGERY_DIR"

# ------------------------------------------------------------------
# 3. Install Docker
# ------------------------------------------------------------------
echo "[3/7] Installing Docker..."
if command -v docker &>/dev/null; then
    echo "  Docker is already installed: $(docker --version)"
else
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER
    echo "  Docker installed. You may need to log out/in for group changes."
fi

# ------------------------------------------------------------------
# 4. Clone repo (or update if exists)
# ------------------------------------------------------------------
echo "[4/7] Setting up project..."
PROJECT_DIR="$HOME/agripay"
if [ -d "$PROJECT_DIR" ]; then
    echo "  Project directory exists. Pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    echo "  Enter the Git repository URL:"
    read -p "  > " GIT_URL
    git clone "$GIT_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi
cd "$PROJECT_DIR/ml-models"

# ------------------------------------------------------------------
# 5. Configure environment
# ------------------------------------------------------------------
echo "[5/7] Configuring environment..."
if [ ! -f .env ]; then
    cp .env.production.example .env
    echo ""
    echo "  !! EDIT .env with your actual credentials !!"
    echo "  Required:"
    echo "    SUPABASE_URL=https://your-project.supabase.co"
    echo "    SUPABASE_SERVICE_ROLE_KEY=your-key"
    echo "    SUPABASE_ANON_KEY=your-key"
    echo "    CORS_ORIGINS=https://agripay.netlify.app"
    echo ""
    read -p "  Press Enter once .env is configured..."
fi

# ------------------------------------------------------------------
# 6. Build and start Docker containers
# ------------------------------------------------------------------
echo "[6/7] Building and starting Docker containers..."
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
echo "  Waiting for services to start..."
sleep 10

# Check health
echo "  Checking service health..."
if curl -sf http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "  ✓ Compute backend is healthy"
else
    echo "  ✗ Compute backend not responding yet (might still be starting)"
fi

if curl -sf http://127.0.0.1:8000/healthz > /dev/null 2>&1; then
    echo "  ✓ TiTiler is healthy"
else
    echo "  ✗ TiTiler not responding yet"
fi

# ------------------------------------------------------------------
# 7. Configure nginx
# ------------------------------------------------------------------
echo "[7/7] Configuring nginx..."
sudo cp deploy/nginx/agripay-compute.conf /etc/nginx/sites-available/agripay-compute
sudo ln -sf /etc/nginx/sites-available/agripay-compute /etc/nginx/sites-enabled/

# Remove default site if present
sudo rm -f /etc/nginx/sites-enabled/default

if sudo nginx -t 2>&1; then
    sudo systemctl reload nginx
    echo "  ✓ nginx configured and reloaded"
else
    echo "  ✗ nginx configuration test failed — check the config"
fi

# ------------------------------------------------------------------
# 8. Configure firewall (optional)
# ------------------------------------------------------------------
echo ""
echo "  Optional: Configure UFW firewall"
echo "    sudo ufw allow ssh"
echo "    sudo ufw allow 8080/tcp    # nginx (for Cloudflare Tunnel)"
echo "    sudo ufw enable"
echo ""

# ------------------------------------------------------------------
# Done!
# ------------------------------------------------------------------
echo "========================================="
echo " Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Install Cloudflare Tunnel (cloudflared):"
echo "     curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
echo "     sudo dpkg -i cloudflared.deb"
echo "     cloudflared tunnel login"
echo "     cloudflared tunnel create agripay"
echo "     cloudflared tunnel route dns agripay compute.yourdomain.com"
echo ""
echo "  2. Create tunnel config (~/.cloudflared/config.yml):"
echo "     tunnel: <TUNNEL-ID>"
echo "     credentials-file: ~/.cloudflared/<TUNNEL-ID>.json"
echo "     ingress:"
echo "       - hostname: compute.yourdomain.com"
echo "         service: http://localhost:8080"
echo "       - service: http_status:404"
echo ""
echo "  3. Start tunnel: cloudflared tunnel run agripay"
echo "     (or install as systemd service: sudo cloudflared service install)"
echo ""
echo "  4. Update Netlify env vars:"
echo "     VITE_COMPUTE_API_URL=https://compute.yourdomain.com"
echo "     VITE_TITILER_URL=https://compute.yourdomain.com"
echo ""
echo "  5. Trigger Netlify redeploy"
echo ""
