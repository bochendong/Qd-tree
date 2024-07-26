import timm

def get_vit_model(model_type, num_classes):
    if (model_type == 'vit_base_patch16_224'):
        return timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    
