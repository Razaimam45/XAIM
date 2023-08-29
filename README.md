# XAIM
Adversarial Detection using Saliency Maps on ViTs

Fine -tuned ViT is avialable at: /l/users/raza.imam/vit_base_patch16_224_in21k_test-accuracy_0.96_chest.pth
Chest X-ray TB Dataset is available at: /l/users/raza.imam/TB_data/
If required for testing: PGD Attack samples is at:  /l/users/raza.imam/attack_images/Test_attacks_PGD/

Also, for loading the fine-tuned ViT, please intall following requirements:
pip install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.11.3+cu102 -f https://download.pytorch.org/whl/torch_stable.html
