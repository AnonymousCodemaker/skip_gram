import random


class skip_gram_dataset():
    def __init__(self, data):
        self.data = data

    @staticmethod
    def load_data(data_path):
        data = []
        with open(data_path, encoding="utf-8") as f:
            sentences = f.readlines()
            for sentence in sentences:
                sentence = sentence.strip().split()
                data.append(sentence)
        return data

    @staticmethod
    def skip_gram(data, vocab, window_size):
        final_data = []
        temp = []
        for sentence in data:
            for i in range(len(sentence)):
                input = vocab.token2id[sentence[i]]
                for left_token in [vocab.token2id[token] for token in sentence[max(i - window_size, 0): i]]:
                    if (input, left_token) not in temp:
                        data_dict = {}
                        data_dict["input"] = (input, left_token)
                        data_dict["label"] = 1
                        final_data.append(data_dict)
                        temp.append((input, left_token))
                for right_token in [vocab.token2id[token] for token in sentence[i + 1:i + 1 + window_size]]:
                    if (input, right_token) not in temp:
                        data_dict = {}
                        data_dict["input"] = (input, right_token)
                        data_dict["label"] = 1
                        final_data.append(data_dict)
                        temp.append((input, right_token))
        for i in range(len(final_data)):
            data_dict = {}
            input1 = random.randint(0, len(vocab) - 1)
            input2 = random.randint(0, len(vocab) - 1)
            while input1 == input2 or (input1, input2) in temp:
                input1 = random.randint(0, len(vocab) - 1)
                input2 = random.randint(0, len(vocab) - 1)
            data_dict["input"] = (input1, input2)
            data_dict["label"] = 0
            final_data.append(data_dict)
            temp.append((input1, input2))

        return final_data

    @staticmethod
    def split_data(data, dev_num, test_num):
        train_data, dev_data, test_data = [], [], []
        dev_list = random.sample(range(0, len(data)), dev_num + test_num)
        for i in range(len(data)):
            if i in dev_list:
                dev_data.append(data[i])
            else:
                train_data.append(data[i])
        test_data = dev_data[dev_num:]
        dev_data = dev_data[0:dev_num]
        return train_data, dev_data, test_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

