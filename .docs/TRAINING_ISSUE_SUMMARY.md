# Training Issue Summary - Replica 2, 3, 5, 6

## Problem
- **Replica 1**: ✅ Normal images (Mean: 138, Std: 57) + Valid meshes (80MB)
- **Replica 2, 3, 5, 6**: ⚠️ Dark images + Corrupted meshes (220 bytes)

## Image Analysis
| Replica | Mean Brightness | Std Dev | Status | Unique Colors |
|---------|----------------|---------|--------|---------------|
| 1 | 138.04 | 56.75 | Normal | 56,000+ |
| 2 | 74.07 | 81.61 | Dark | 29,118 |
| 3 | 53.53 | 60.75 | Darker | 27,677 |
| 5 | 21.08 | 36.77 | Very Dark | 6,554 |
| 6 | 59.36 | 79.90 | Dark | 36,495 |

## Mesh Analysis
- **Replica 1, 4, 7, 8**: Valid meshes (45-80 MB)
- **Replica 2, 3, 5, 6**: Corrupted meshes (220 bytes - empty/corrupted)

## Root Cause Hypothesis
1. **Training interrupted early** - Model didn't learn scene properly
2. **GPU memory issues** - Training crashed during mesh extraction
3. **Disk space issues** - Mesh export failed
4. **Dataset issues** - Missing or corrupted data for these scans

## Recommendations
1. **Check training logs** in `runs_new/replica_X/2025-XX-XX_XX-XX-XX/logs/` for errors
2. **Verify dataset** - Check if `data/Replica/scan2/`, `scan3/`, `scan5/`, `scan6/` have all required files
3. **Re-run training** for replicas 2, 3, 5, 6
4. **Check system resources** - Monitor GPU memory and disk space during training

## Evaluation Status
- ✅ **Replica 1** (room0): Can be evaluated
- ❌ **Replica 2** (room1): Cannot evaluate - no valid meshes
- ⚠️ **Replica 3** (room2): Can evaluate with `mesh_240.ply` (not mesh_2400.ply)
- ❌ **Replica 5** (office1): Cannot evaluate - no valid meshes
- ❌ **Replica 6** (office2): Cannot evaluate - no valid meshes

## Next Steps
1. Investigate why training failed for replicas 2, 3, 5, 6
2. Re-run training for these replicas
3. Use `check_mesh_files.py` to verify mesh validity before evaluation
4. Use `diagnose_image_issue.py` to check image quality
