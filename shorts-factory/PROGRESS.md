Task: predixel-night-3d (27 Nov 00:30 run): build + render ONE 3D Short in Blender (no upload)
Done:
- [x] git pull, strategy + playbook read
- [x] blender/wreckingwall3d.py (CRUSH: wrecking ball, 4 swings 22/38/56/80 deg = 12/20/29/40 km/h, 172-brick wall) and blender/post_wreckingwall.py written; swing schedule verified outside Blender
Next: next night run (or any free slot): free RAM check, then `D:\tools\blender-4.5\blender.exe -b -P blender/wreckingwall3d.py -- out/3d_wreckingwall 750 16 1`, then `py blender/post_wreckingwall.py out/3d_wreckingwall`, qa.py, mkupload.py, contact sheet, PLAYBOOK line, commit/push
Status: STOPPED, nothing rendered: free RAM stayed under 1.5 GB (1.16 / 1.32 / 1.41 / 1.10 GB over 30 min of retries; 16 GB total, used by Chrome, ChatGPT, Claude, Defender and QFlow). Scripts are committed and ready; no owner action needed unless RAM stays this low every night (closing ChatGPT/Chrome at night would free it).

Previous (26 Nov daily run): DONE. Next daily run: pins once public (Crush Count, Merge Form, Ice Crush, Balloon Lift; texts in PLAYBOOK); build 2 more (SCALE + CRUSH, new objects) for 28 Nov; package tonight's 3D Short (out/wreckingwall_3d.mp4) for the next free 19:00 slot
