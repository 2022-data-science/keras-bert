import csv
import tensorflow as tf
import transformers
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
labels = ["contradiction", "entailment", "neutral"]
max_length = 64  # Maximum length of input sentence to the model.
batch_size = 8
epochs = 3

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.
    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
            self,
            sentence_pairs,
            labels,
            batch_size=batch_size,
            shuffle=True,
            include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        self.tokenizer = pre_tokenizer
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        # print(indexes)
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


print("???????????????...")
model = tf.keras.models.load_model("./model", custom_objects=None, compile=True, options=None)
print("Tokenizer?????????...")
pre_tokenizer = transformers.BertTokenizer.from_pretrained(
            "bert-base-chinese", do_lower_case=True)
# tmp_sentence = np.array([[str("1"), str("2")]])
# tmp_test_data = BertSemanticDataGenerator(
#     tmp_sentence, labels=None, batch_size=1, shuffle=False, include_targets=False,
# )

print("??????????????????...")
with open("./meta_data.csv", encoding='utf-8') as f:
    reader = csv.reader(f)
    meta_list = []
    for i in reader:
        meta_list.append(i[0])
sentences = np.array([["", i] for i in meta_list])


def check_similarities(sentence1):
    for s in sentences:
        s[0] = sentence1
    test_data = BertSemanticDataGenerator(
        sentences, labels=None, batch_size=sentences.size, shuffle=False, include_targets=False,
    )

    proba_list_np = model.predict(test_data[0], batch_size=600, use_multiprocessing=True)

    proba_list = proba_list_np.tolist()
    t = 0
    for k in proba_list:
        k.append(t)
        t += 1

    proba_list.sort(key=lambda x: x[1], reverse=True)
    return proba_list


def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .6f}"
    pred = labels[idx]
    return pred, proba


def show_meta():
    for t in meta_list:
        print(t)


def add_meta(meta):
    if meta in meta_list:
        print("??????????????????")
        return
    meta_list.append(meta)
    with open("./meta_data.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for r in meta_list:
            writer.writerow([r])
        print("success")


def del_meta(meta):
    if not meta in meta_list:
        print("??????????????????")
        return
    meta_list.pop(meta_list.index(meta))
    with open("./meta_data.csv", 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for r in meta_list:
            writer.writerow([r])
    print("success")


if __name__ == "__main__":
    print("????????????????????????????????????????????????\n"
          '??????????????????(???" "??????)???????????????\n'
          '??????"meta"?????????????????????\n'
          '??????"add + ?????????"???????????????\n'
          '??????"del + ?????????"???????????????\n')
    while True:
        inputs = input("???????????????").split()
        if len(inputs) == 1:
            if inputs[0] == "meta":
                show_meta()
            else:
                res = check_similarities(inputs[0])[0:5]
                for i in res:
                    print(meta_list[i[3]], i[1])
        if len(inputs) == 2:
            if inputs[0] == "add":
                add_meta(inputs[1])
            elif inputs[0] == "del":
                del_meta(inputs[1])
            else:
                res = check_similarity(inputs[0], inputs[1])
                print("????????????: " + res[0] + " ??????: " + res[1])

