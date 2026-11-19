# From coursework to repository

This repository started as coursework for SCC456 Generative AI: Theory and Applications at Lancaster University. The original consisted of four training scripts, one per phase, an interactive generation script, and a ten-page report, kept as submitted in [coursework_report.pdf](coursework_report.pdf). This page records what the rebuild changed, the problems it found in the original code, and whether the report's conclusions hold up under the new evaluation.

## The original project

| Script | What it did |
| --- | --- |
| `phase1_baseline.py` | Computed the mean training image of each digit and its pixel MSE on the test split |
| `phase2_vanilla_gan.py` | Trained the fully connected GAN for 50 epochs, saved samples, the loss curve, and the generator, and printed its pixel MSE |
| `phase3_dcgan.py` | The same for the convolutional GAN |
| `phase4_cgan.py` | The same for the conditional GAN |
| `generate.py` | Asked for a digit in the terminal, generated one image, and saved it |

Models were compared with pixel MSE against the first 1,000 test images and by looking at generated samples. The report gave these results:

| Model | Pixel MSE | Generates a chosen digit |
| --- | --- | --- |
| Baseline | 0.2117 | Yes, always the same image |
| Vanilla GAN | 0.5119 | No |
| DCGAN | 0.5212 | No |
| Conditional GAN | 0.4129 | Yes |

## Problems found in the original code

### The DCGAN produced 24x24 images

The last transposed convolution of the DCGAN generator used padding 3. A transposed convolution with kernel 4 and stride 2 turns a 14x14 input into (14 - 1) x 2 - 2 x padding + 4 pixels: 28 with padding 1, but 24 with padding 3.

The research log describes a shape mismatch during DCGAN training, solved by adding `AdaptiveAvgPool2d((7, 7))` to the discriminator. That let the code run, but the cause was in the generator. Real images reached the discriminator at 28x28 and generated ones at 24x24, and the pooling layer stretched the feature maps of the generated images from 6x6 to 7x7 to hide the difference. A difference in size is easy for a discriminator to detect, which may explain the collapse of the discriminator loss to zero that the research log reports.

The script's MSE step subtracts a 28x28 test image from a 24x24 generated one, which raises a shape error, so the script stops before printing an MSE. The DCGAN value in the report must come from a different version of the code. The saved coursework weights also produce 24x24 images.

The conditional GAN, written afterwards, uses padding 1 and produces 28x28 images. It kept the pooling layer, which then does nothing.

### Label smoothing was not where the report says

The report says that label smoothing, a target of 0.9 instead of 1.0 for real images, was applied to the DCGAN and the conditional GAN, and that it fixed the DCGAN's discriminator collapse. In the submitted scripts only the conditional GAN uses it. The vanilla GAN and the DCGAN train with a target of 1.0.

### The last step of the ladder changed three things

Between the DCGAN and the conditional GAN, the code added conditioning, fixed the output size, and added label smoothing. The report credits the improvement to conditioning, but the original code cannot separate the three changes.

### The baseline was scored on different test images

`phase1_baseline.py` averages the error over all 10,000 test images, while the GAN scripts use the first 1,000. On all 10,000 the baseline scores 0.2117, exactly as reported, and on the first 1,000 it scores 0.2089. The difference is small, but the comparison was not like for like.

### Each result came from a single run

Every script set seed 42 once at the start, so the noise used for samples and MSE depended on everything drawn during training. With one run per model there was no way to tell how much a result would change with another seed.

## What changed

| Area | Coursework | This repository |
| --- | --- | --- |
| Code | Five scripts, with networks and the training loop copied between them | A typed package with one training loop for every model. The conditional GAN is the DCGAN built with `conditional=True`. |
| DCGAN output | 24x24, hidden by a pooling layer | 28x28, no pooling layer |
| Label smoothing | Conditional GAN only | All three GANs, one-sided at 0.9 |
| Loss | Sigmoid followed by `BCELoss` | `BCEWithLogitsLoss` on logits: the same objective, computed in a numerically stable way |
| Runs | One per model | Three per model, seeds 42, 43, and 44, reported as mean and standard deviation |
| Evaluation noise | Whatever the global random state held after training | A fixed evaluation seed, the same noise for every model |
| Metrics | Pixel MSE and visual inspection | Pixel MSE, FID, recognisable rate, class balance, and correct-digit rate, with real images as reference points |
| Pixel MSE images | All 10,000 test images for the baseline, the first 1,000 for the GANs | The first 1,000 for every source |
| Generation | A terminal prompt that writes one digit per image | A CLI and an HTTP API that write whole numbers in a consistent style, reproducible from a seed |
| Checkpoints | Bare weights, with the generator class copied into `generate.py` | Weights with the model settings, training history, and library versions, loaded with `weights_only=True` |
| Engineering | None | Tests, ruff, strict mypy, pre-commit, CI, and a Docker image |

Everything else stayed as it was: the MNIST split, scaling pixels to [-1, 1], the layer sizes of every network, 50 epochs, batch size 128, Adam with learning rate 0.0002 and betas 0.5 and 0.999, 100 noise values, and pixel MSE on the first 1,000 test images.

## Pixel MSE, then and now

| Model | Report | This repository, mean of three seeds |
| --- | --- | --- |
| Baseline | 0.2117 on all test images | 0.209 on the first 1,000, no training |
| Vanilla GAN | 0.5119 | 0.516 ± 0.007 |
| DCGAN | 0.5212 | 0.527 ± 0.006 |
| Conditional GAN | 0.4129 | 0.419 ± 0.005 |

The values reproduce closely. The rebuild adds what the report could not show: an independent real image of the same digit scores 0.418 and a real image of any digit 0.520. Every GAN scores about what real handwriting scores.

## Do the conclusions hold?

| Claim in the report | Verdict | Evidence |
| --- | --- | --- |
| The conditional GAN performed best of the three GANs | Holds, and much more clearly | FID 7.4 ± 0.8, against 62.5 ± 9.7 for the vanilla GAN and 64.1 ± 4.4 for the DCGAN |
| The baseline has the lowest MSE despite the worst images, so MSE is a weak metric | Holds, and the reason is now known | The class mean minimises the expected squared error. A real image of the same digit scores about twice as much, 0.418 against 0.209. |
| The conditional GAN's MSE of 0.4129 beats the vanilla GAN's and the DCGAN's | The numbers reproduce, but they do not show this | The gap comes from pairing. The conditional GAN is compared with test images of the requested digit and the others with any digit, and real images show the same gap, 0.418 against 0.520. |
| The DCGAN draws sharper, cleaner digits than the vanilla GAN | Partly | Its strokes are smoother, but in every seed fewer of its digits are recognisable, 67% against 80% on average, and its FID is no better |
| Batch normalisation made DCGAN training more stable | Consistent, not tested | The DCGAN's losses settle within ten epochs while the vanilla GAN's generator loss spikes, but no run isolates batch normalisation, and steady losses did not lead to better samples |
| Each model addresses a limitation of the one before | Not at every step | The DCGAN step traded recognisable digits, down from 80% to 67%, for better class balance, up from 0.972 to 0.987, and left FID within the spread between seeds |
| The conditional GAN generates all ten digits on request | Holds | 99.0% of samples are classified as the requested digit, averaged over three seeds, and at least 97% of every digit in every run |

Two items from the report's future work are now done: FID, computed in the feature space of an MNIST classifier, and a user-facing application, the API. Diffusion models and higher-resolution datasets remain open.
