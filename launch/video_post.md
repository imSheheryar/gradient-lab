# Video post — follow-up to the carousel

Post this three or four days after the document post so the two don't compete in the same
feed cycle. Upload `launch/gradient_lab_walkthrough.mp4` as a native video (1080 × 1080,
29.6 s, H.264). LinkedIn autoplays muted, so the video carries its own captions.

## Caption

**𝐓𝐡𝐢𝐫𝐭𝐲 𝐬𝐞𝐜𝐨𝐧𝐝𝐬 𝐨𝐧 𝐰𝐡𝐞𝐫𝐞 𝐝𝐞𝐩𝐭𝐡 𝐬𝐭𝐨𝐩𝐬 𝐛𝐞𝐢𝐧𝐠 𝐟𝐫𝐞𝐞.**

The carousel earlier this week had the numbers. This is the same result with the camera
moving: eight slides, no narration, because the figures say it better than I can.

Depth 8 → 32, a plain sigmoid network hands its first layer 1.1e-17 of the gradient. One
residual connection and the same network finishes at 0.994. Sixteen times the width does
nothing at all.

Code, the 720-row CSV and every figure: https://github.com/imSheheryar/gradient-lab

#MachineLearning #DeepLearning #NumPy #Autograd #Python #AIResearch #BuildInPublic

## Notes

- Square video is the safest crop on both mobile and desktop feed.
- The first frame is a title card, so the muted autoplay still reads as a poster.
- If you would rather post a screen recording instead, record the probe demo:
  `python experiments/probe_demo.py` writes the trajectory table, and
  `figures/probe_trajectories.png` is the frame worth showing.
