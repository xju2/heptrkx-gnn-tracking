"""Take a model configation and evaluate hitsgraph"""

from models import get_model

def load_model(config, reload_epoch):
    model_config = config['model']
    model_type = model_config.pop('model_type')
    model_config.pop('optimizer', None)
    model_config.pop('learning_rate', None)
    model_config.pop('loss_func', None)
    model = get_model(name=model_type, **model_config)

    # Reload specified model checkpoint
    output_dir = get_output_dir(config)
    checkpoint_file = os.path.join(output_dir, 'checkpoints',
                                   'model_checkpoint_%03i.pth.tar' % reload_epoch)
    model.load_state_dict(torch.load(checkpoint_file, map_location='cpu')['model'])
    return model


def create_evaluator(train_config_file, reload_epoch):
    pass
