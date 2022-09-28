import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel


class CustomModel(nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.num_labels = 56
        self.model_name = model_name

        self.model = ViTModel.from_pretrained(self.model_name)

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(0.2)
        self.fc_1 = nn.Linear(1024, 512)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(0.1)
        self.fc_2 = nn.Linear(512, self.num_labels)

    def forward(self, pixel_values, labels):
        # Extract outputs from the body
        outputs = self.model(pixel_values)

        output = self.relu_1(outputs.pooler_output)
        output = self.fc_1(self.dropout_1(output))
        output = self.relu_2(output)
        output = self.fc_2(self.dropout_2(output))
        logits = output

        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='sum')

        return {"loss": loss, "logits": logits, "hidden_states": outputs.hidden_states, "pooler_output": outputs.pooler_output}
