# Model card: polyp-segmentation-unet

## Model details
- **Architecture:** U-Net (`segmentation_models_pytorch`) with an ImageNet-pretrained ResNet-34 encoder, `Dropout2d(0.2)` after each of the five decoder blocks, one sigmoid output channel.
- **Input:** RGB colonoscopy frame, resized to 320×320, ImageNet-normalised. Output mask is resized back to the original resolution.
- **Training:** 800 Kvasir-SEG images, BCE + soft Dice loss, AdamW (lr 3e-4, wd 1e-4), OneCycle schedule, batch 8, 40 epochs, flips / rot90 / affine / colour / blur augmentation. Best validation-Dice checkpoint (epoch 35). Trained on an Apple M5 GPU (MPS) in 40 minutes (retrained 2026-10-10 after fixing per-worker augmentation seeding). Full config: [`configs/kvasir_unet.yaml`](configs/kvasir_unet.yaml).
- **Author:** Muhammad Haroon Khan. **License:** MIT (code). Dataset: CC BY 4.0.

## Intended use
A portfolio / research demonstration of medical image segmentation, uncertainty estimation and honest evaluation. Suitable for learning, benchmarking ideas on Kvasir-SEG and as a starting point for further research.

**Out of scope:** any clinical, diagnostic or screening use; any use on patients; use on data from other endoscopes, hospitals or populations without a new validation study. **This is not a medical device and has no regulatory clearance.**

## Data
Kvasir-SEG (Jha et al., 2020): 1,000 colonoscopy frames with polyp masks from Vestre Viken Health Trust, Norway, resolutions 332×487 to 1920×1072. Split with a fixed seed into 800 train / 100 val / 100 test ([`configs/split.json`](configs/split.json)). The masks are JPEGs and were binarised at 127.

## Evaluation (test split, 100 images, threshold 0.5, original resolution)

| Metric | Deterministic | MC-dropout mean (20 passes) |
|---|---|---|
| mean per-image Dice | 0.904 | 0.904 |
| mean per-image IoU | 0.844 | 0.845 |
| precision / recall | 0.937 / 0.901 | 0.937 / 0.903 |
| dataset-level (pooled) Dice | 0.903 | 0.904 |

By polyp size (mean Dice): small (<5%) 0.868 (n=16), medium (5-20%) 0.938 (n=51), large (>20%) 0.867 (n=33).

Uncertainty: std on mis-classified pixels 0.103 vs. 0.010 on correct pixels; pixel-error AUROC 0.930; Spearman(image Dice, mean std) = -0.36.

## Limitations and risks
- **Single source.** One hospital, one family of endoscopes, 1,000 frames. Performance on other equipment, lighting, bowel preparation quality or patient populations is unknown and will likely be worse.
- **No negatives.** Every training and test frame contains a polyp, so the false-positive rate on healthy mucosa was never measured. The model may highlight folds, bubbles or specular reflections.
- **Small and large polyps are the weakest categories** (Dice 0.868 and 0.867 vs. 0.938 for medium; see the size table). Small polyps are also the clinically most important ones, and with only 16 small polyps in the test set their score is the least certain.
- **Small test set.** 100 images give roughly ±0.025 uncertainty on mean Dice (95 % interval); differences of that size between settings are not meaningful.
- **Uncertainty is a heuristic.** MC dropout captures model (epistemic) uncertainty only, under an approximation; a confident-looking map does not imply a correct mask.
- **Labels are imperfect.** Some Kvasir-SEG masks are coarse polygons; the model is scored against them, so part of the "error" is annotation noise.

## Ethical considerations
The dataset is anonymised and publicly released by its authors. No patient data was collected for this project. Any downstream use on real patients would require prospective validation, regulatory approval and clinician oversight; a tool like this could at most *assist* an endoscopist, never replace one.
