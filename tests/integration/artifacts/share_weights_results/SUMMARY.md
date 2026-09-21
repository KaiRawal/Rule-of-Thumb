# Experiment 1 - Digits, 10-class shared vs unshared RoT weights

N_train=1437  N_test=360  majority=0.102  stratified 80/20, seed 0

| backbone | features | sh-train | sh-test | sh-params | un-train | un-test | un-params | test delta |
|---|---|---|---|---|---|---|---|---|
| raw | 1x8x8 | 0.1176 | 0.1167 | 30 | 0.9694 | 0.9750 | 1,290 | +0.8583 |
| coords | 3x8x8 | 0.3633 | 0.3500 | 70 | 0.9722 | 0.9750 | 3,850 | +0.6250 |
| mobilenet_v3_small | 576x7x7 | 1.0000 | 0.9667 | 11,530 | 1.0000 | 0.9889 | 564,490 | +0.0222 |
| resnet18 | 512x7x7 | 1.0000 | 0.9667 | 10,250 | 1.0000 | 0.9944 | 501,770 | +0.0277 |
| convnext_tiny | 768x7x7 | 1.0000 | 0.9639 | 15,370 | 1.0000 | 0.9889 | 752,650 | +0.0250 |
| efficientnet_b0 | 1280x7x7 | 0.9993 | 0.9611 | 25,610 | 1.0000 | 0.9944 | 1,254,410 | +0.0333 |
