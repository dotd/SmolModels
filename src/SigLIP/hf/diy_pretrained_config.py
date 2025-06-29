from transformers import SiglipConfig, SiglipModel

# Initializing a SiglipConfig with google/siglip-base-patch16-224 style configuration
configuration = SiglipConfig()

# Initializing a SiglipModel (with random weights) from the google/siglip-base-patch16-224 style configuration
model = SiglipModel(configuration)

# Accessing the model configuration
configuration = model.config

# We can also initialize a SiglipConfig from a SiglipTextConfig and a SiglipVisionConfig
from transformers import SiglipTextConfig, SiglipVisionConfig

# Initializing a SiglipText and SiglipVision configuration
config_text = SiglipTextConfig()
config_vision = SiglipVisionConfig()

config = SiglipConfig.from_text_vision_configs(config_text, config_vision)
