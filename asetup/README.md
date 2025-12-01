# ND-Uncertainty Setup

## Local Machine (3090)

1. Activate environment:
   ```bash
   conda activate nd_unc
   ```

2. Install packages:
   ```bash
   cd asetup
   pip install -r requirements.txt
   ```

   Or use the batch file:
   ```bash
   setup_local.bat
   ```

3. Test:
   ```bash
   python asetup/test_setup.py
   ```

## Workstation (A100 with CUDA 12)

1. Activate existing `nd_unc` environment
2. Run:
   ```bash
   cd asetup
   chmod +x update_nd_unc.sh
   ./update_nd_unc.sh
   ```

That's it. DINOv2 loads automatically via torch.hub.
