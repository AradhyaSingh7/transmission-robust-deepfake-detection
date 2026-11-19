# Evaluation Samples

Curated sample face images for rapid local testing and inference verification.

### Organization
- `samples/real/`: Authentic facial photographs sourced from the FFHQ dataset.
- `samples/fake/`: Synthetic facial portraits synthesized by StyleGAN generators.

### Usage
Run inference on any individual sample:
```bash
python demo.py --image samples/fake/fake_01.jpg
```
Or test under transmission compression:
```bash
python demo.py --image samples/real/real_01.jpg --degrade jpeg50
```
