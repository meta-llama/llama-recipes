# Reference code :https://github.com/yangjianxin1/Firefly/blob/2cbefd968391e024592fb89e058929cf118af071/component/dataset.py#L6
import json
from torch.utils.data import Dataset

SYSTEM_FORMAT = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>\n'
USER_FORMAT = '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
ASSISTANT_FORMAT = '{content}<|eot_id|>\n'


class ShareGPT_Dataset(Dataset):

    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.tokenizer = tokenizer
        self.system_format = SYSTEM_FORMAT
        self.user_format = USER_FORMAT
        self.assistant_format = ASSISTANT_FORMAT
        self.defaults_system = None

        with open(dataset_config.data_path, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        self.data_list = data_list

        if partition == "train":
            self.data_list = self.data_list[200:]
        else:
            self.data_list = self.data_list[:200]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        IGNORE_INDEX = -100
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, labels = [], []

        if self.system_format is not None:
            system = data['system'].strip() if 'system' in data.keys() else self.defaults_system

            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                labels = [IGNORE_INDEX] * len(input_ids)

        conversations = data['conversation']

        for i, conv in enumerate(conversations):
            human = conv.get('human', "").strip()
            assistant = conv.get('assistant', "").strip()

            human = self.user_format.format(content=human, stop_token=self.tokenizer.eos_token)
            assistant = self.assistant_format.format(content=assistant, stop_token=self.tokenizer.eos_token)

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            labels += [IGNORE_INDEX] * len(input_tokens) + output_tokens

        assert len(input_ids) == len(labels)

        input_ids.pop()
        labels.pop()
        input_ids[-1] = self.tokenizer.eos_token_id
        labels[-1] = self.tokenizer.eos_token_id

        attention_mask = [True] * len(input_ids)
        assert len(input_ids) == len(labels) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }
        return inputs
