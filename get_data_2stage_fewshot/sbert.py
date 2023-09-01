import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import faiss


class SentRetriever(object):
    def __init__(
            self,
            output_sents_file="/sharefs/guozhicheng/FiD/get_data_sbert/wiki1m.simcse.features",
            ext_sents=[],
            dataset_sents=[],
            s_bert='princeton-nlp/unsup-simcse-bert-base-uncased',
            index_to_gpu=False
    ):
        self.output_sents_file = output_sents_file
        self.batch_size = 128
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(s_bert)
        self.model = AutoModel.from_pretrained(s_bert).to(self.device)

        if os.path.isfile(self.output_sents_file):
            self.sent_features, self.sent_set = self.load_sent_features(self.output_sents_file)
            print(f"{len(self.sent_set)} sents loaded from {self.output_sents_file}")
        else:
            self.sent_set = ext_sents
            self.sent_features = self.build_sent_features(self.sent_set)
            self.save_sent_features(self.sent_features, self.sent_set, self.output_sents_file)

        self.dataset_sent_features = self.build_sent_features(dataset_sents)
        self.all_feature = torch.cat([self.dataset_sent_features, self.sent_features])
        self.all_sents = dataset_sents + self.sent_set
        self.index_to_gpu = index_to_gpu

    def build_sent_features(self, text_sets):
        print(f"Build features for {len(text_sets)} sents...")
        batch_size, counter = self.batch_size, 0
        batch_text = []
        all_i_features = []
        # Prepare the inputs
        for i_n in tqdm(text_sets):
            counter += 1
            batch_text.append(i_n)
            if counter % batch_size == 0 or counter >= len(text_sets):
                with torch.no_grad():
                    # print(batch_text)
                    i_input = self.tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True,
                                             max_length=512).to(self.device)
                    i_feature = self.model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
                    # i_feature = self.model(**i_input, output_hidden_states=True)
                    # print(i_feature[0].size())
                    # print(i_feature[1].size())
                i_feature /= i_feature.norm(dim=-1, keepdim=True)
                all_i_features.append(i_feature.squeeze().to('cpu'))
                batch_text = []
        returned_text_features = torch.cat(all_i_features)
        return returned_text_features

    def save_sent_features(self, sent_feats, sent_names, path_to_save):
        assert len(sent_feats) == len(sent_names)
        print(f"Save {len(sent_names)} sent features at {path_to_save}...")
        torch.save({'sent_feats': sent_feats, 'sent_names': sent_names}, path_to_save)
        print(f"Done.")

    def load_sent_features(self, path_to_save):
        print(f"Load sent features from {path_to_save}...")
        checkpoint = torch.load(path_to_save)
        return checkpoint['sent_feats'], checkpoint['sent_names']

    def get_text_features(self, text):
        with torch.no_grad():
            i_input = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device)
            text_features = self.model(**i_input, output_hidden_states=True, return_dict=True).pooler_output
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def retrieve(self, text):
        text_f = self.get_text_features(text)
        similarity, ret_idx = self.single_dot_retrieval(text, text_f, self.sent_features)
        return similarity, ret_idx

    def single_dot_retrieval(self, text, text_f, sent_f, neighbors=100):
        # Pick the top 5 most similar labels for the image
        N = 50000
        n_split = (len(sent_f) // N) + 1
        similarity = []
        for i in range(n_split):
            i_f = sent_f[i * N: (i + 1) * N].to(self.device) if (i + 1) * N < len(
                sent_f) else sent_f[i * N:].to(self.device)
            similarity.append((100 * i_f @ text_f.T).T.softmax(dim=-1).squeeze())

        similarity = torch.cat(similarity)
        ret_idx = []
        values, indices = similarity.topk(neighbors) if neighbors >= 5 else similarity.topk(len(self.sent_set))
        # Print the result
        # print(f"\nTop predictions of '{text}':\n")
        for value, index in zip(values, indices):
            # print(f"{index} {self.sent_set[index]}:  {100 * value.item():.2f}%")
            ret_idx.append(index.cpu().detach().numpy().tolist())
        return values, ret_idx

    def setup_faiss(self, device=0):
        d = self.all_feature.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.all_feature.numpy())
        if self.index_to_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            # flat_config = faiss.GpuIndexFlatConfig()
            # flat_config.device = 0
            self.index = faiss.index_cpu_to_gpu(res, device, self.index)

    def faiss_retrieval(self, text, num_samples=50):
        text_f = self.get_text_features(text).cpu().numpy()
        distance, idx = self.index.search(text_f, num_samples)
        return distance, idx

    @staticmethod
    def retrieve_in_batch(retriever, all_features, batch_size=16):
        counter = retriever.batch_size
        batch_text = []
        distances, indices = [], []
        # Prepare the inputs
        for i_n in tqdm(all_features):
            counter += 1
            batch_text.append(i_n)
            if counter % batch_size == 0 or counter >= len(all_features):
                dist, idx = retriever.index.search(np.stack(batch_text), 50)
                distances.append(dist)
                indices.append(idx)
                batch_text = []
        distances = np.concatenate(distances, 0)
        indices = np.concatenate(indices, 0)
        return distances, indices


def load_file(filename):
    i_set = []
    with open(filename, 'r') as f:
        for i_n in f:
            i_set.append(i_n.strip('\n'))
    return i_set



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    all_sents = load_file('/sharefs/guozhicheng/FiD/wiki1m_for_simcse.txt')
    sentRetriever = SentRetriever(output_sents_file="/sharefs/guozhicheng/FiD/get_data_sbert/wiki1m.simcse.features",
                                  ext_sents=all_sents,
                                  s_bert='princeton-nlp/unsup-simcse-bert-base-uncased')

    sentRetriever.retrieve("No reason to watch. It was terrible.")
