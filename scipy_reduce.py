from collections import OrderedDict
import os
from tempfile import gettempdir

import pandas as pd
import spacy

import scipy
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from sentence_transformers import SentenceTransformer

p = 0

class LevelClusterer:
    def __init__(self,
                 dataframe,
                 text_col='cleaned_texts',
                 BERTopic_model=None,
                 embeddings = None
                ):
        self.BERTopic_model = BERTopic_model
        self.embeddings = embeddings
        self.df = dataframe.rename(columns={text_col:'Document'})
        self.initial_rank = []
        self.new_rank = []
        self.distance_matrix = None
        self.linkage_matrix = None
        self.cutree = None
        self.height = None
        self.levels = None
        self.p = 0


    def initial_rank(self):
        axis = "yaxis"
        if self.BERTopic_model is not None:
            fig = self.BERTopic_model.visualize_hierarchy()
        initial_rank = []
        for x in fig.layout[axis]["ticktext"]:
            initial_rank.append(int((x.split('_')[0])))

        self.initial_rank = initial_rank

    def calculate_distance_matrix(self,embeddings=None):
        if self.BERTopic_model is not None:
            embeddings = np.array(self.BERTopic_model.topic_embeddings)
            topics = sorted(list(self.BERTopic_model.get_topics().keys()))
            all_topics = sorted(list(self.BERTopic_model.get_topics().keys()))
            indices = np.array([all_topics.index(topic) for topic in topics])
            embeddings = embeddings[indices]

            num_unique_topics = len(topics)
            level = 0
            while num_unique_topics>1:
                num_unique_topics = num_unique_topics/2
                level += 1

            levels = [i for i in range(level)]
            self.height = levels
        elif self.embeddings is not None:
            embeddings = self.embeddings

            num_unique_topics = embeddings.shape[0]
            level = 0
            while num_unique_topics>1:
                num_unique_topics = num_unique_topics/2
                level += 1

            levels = [i for i in range(level)]
            self.height = levels

        distance_matrix = 1 - cosine_similarity(embeddings)
        self.distance_matrix = distance_matrix
        return distance_matrix
        


    def calculate_cutree(self,distance_matrix=None,height=[0,1,2]):
        if self.distance_matrix is not None:
            pairwise_distance = pdist(self.distance_matrix) 
        else:
            pairwise_distance = pdist(distance_matrix) 

        if self.height is not None:
            height = self.height
        linkage_matrix = ward(pairwise_distance)
        self.linkage_matrix = linkage_matrix
        cutree = scipy.cluster.hierarchy.cut_tree(linkage_matrix, height=height)
        # check for duplicate cuts in cutree
        cuts = []
        for count in height:
            if len(cuts) == 0:
                cuts.append([i[count] for i in cutree])
            elif [i[count] for i in cutree] != cuts[-1]:
                cuts.append([i[count] for i in cutree])

        self.levels = len(cuts)
        cutree = []
        for i in range(0,len(cuts[0])):
            cutree.append([cut[i] for cut in cuts])

        cutree = np.array(cutree)

        self.cutree = cutree
        return cutree

    def cut_at_level(self,level = 1):
        
        self.df['Topic'] = self.df['topic_number_{}'.format(level)]
        topics = self.df.Topic.tolist().copy()
        unique_topics = sorted(list(self.df.topic_number_1.unique()))
        # max_topic = unique_topics

        predictions = [i[level] for i in self.cutree]

        # Map similar topics
        mapped_topics = {unique_topics[index]: prediction
                        for index, prediction in enumerate(predictions)
                        if prediction != -1 and unique_topics[index] != -1}
                        
        self.df['Topic'] = self.df.topic_number_1.map(mapped_topics).fillna(self.df.Topic).astype(int)
        mapped_topics = {from_topic: to_topic for from_topic, to_topic in zip(topics, self.df.Topic.tolist())}

        topics_words = self.get_topics_words()

        combined_topics = {}
        for key,value in mapped_topics.items():
            if value not in combined_topics:
                combined_topics[value]=[key]
            elif key not in combined_topics[value]:
                combined_topics[value].append(key)

        new_rank = []

        for rank in self.initial_rank:
            new_rank.append(mapped_topics[rank])

        new_rank = list(dict.fromkeys(new_rank))
        self.initial_rank = new_rank

        if self.BERTopic_model is not None:
            new_topics_words,new_topics_embeddings = self.new_topics_embeddings_and_words(self,combined_topics,topics_words)

            self.update_BERTopic_model(mapped_topics)
        else:
            new_topics_words,new_topics_embeddings = self.new_topics_embeddings_and_words(self,combined_topics,topics_words)
            self.embeddings = list(new_topics_embeddings.values())

        keys = list(new_topics_embeddings.keys())
        keys.sort()
        ordered_embeddings = dict()

        for key in keys:
            ordered_embeddings[key] = new_topics_embeddings[key]

        new_topics_embeddings = ordered_embeddings

        keys = list(new_topics_words.keys())
        keys.sort()
        ordered_words = dict()

        for key in keys:
            ordered_words[key] = new_topics_words[key]

        new_topics_words = ordered_words

        embedding_list=list(new_topics_embeddings.values())
        self.BERTopic_model.topic_embeddings = embedding_list

        new_model_topics = self.new_model_topic_words(new_topics_words)

        topics = self.topic_names_dict(new_model_topics,num_words=5)

        cluster_names_column = pd.Series(self.df['Topic'].values).apply(lambda x: topics[str(x)])
        self.df['topic_name_{}'.format(str(level+1))] = cluster_names_column.values
        self.df=self.df.sort_values(['topic_number_1'], ascending=True)
        self.df = self.df.rename(columns={'Topic':'topic_number_{}'.format(str(level+1))})
        print(self.df.head(10))

        if self.BERTopic_model is not None:
            global p
            p = len(self.BERTopic_model.topic_embeddings)
        else:
            global p
            p = self.embeddings.shape[0]

        fig = self.plot_dendrogram(level,new_rank,topics)
        fig.write_image(os.path.join(gettempdir(), "dendrogram_level_{}.png".format(str(level+1))))

        return fig, self.df.copy(deep=True)

    def recursive_leveling(self):
        # TODO : recursive leveling
        for level in range(1,self.levels):
            fig,df = self.cut_at_level(level)
        return fig, self.df.copy(deep=True)

    def get_topics_words(self):
        nlp = spacy.load("en_core_web_sm")

        topics_words = {}

        if self.BERTopic_model is not None:
            for key,value in self.BERTopic_model.get_topics().items():
                topic_words = [i[0] for i in value]

                unallowed_postags=['X', 'SYM', 'PUNCT', 'NUM','SPACE','ADP','AUX','CONJ','CCONJ','DET','PRON','SCONJ']
                doc = nlp(' '.join(topic_words))
                words = [token.text for token in doc if token.pos_ not in unallowed_postags and token.lemma_!='-PRON-']
            
                topics_words[key]=words
        return topics_words

    def new_topics_embeddings_and_words(self,combined_topics,topics_words):
        new_topics_words = {}
        new_topics_embeddings = {}

        if self.BERTopic_model is not None:
            for key,value in combined_topics.items():
                words = []
                embeddings = []
                for i in value:
                    words.extend(topics_words[i])
                    embeddings.append(self.BERTopic_model.topic_embeddings[i+1])

                new_topics_words[key] = words
                new_topics_embeddings[key] = np.mean(embeddings, axis=0)
        else:
            for key,value in combined_topics.items():
                words = []
                embeddings = []
                for i in value:
                    words.extend(topics_words[i])
                    embeddings.append(self.embeddings[i])

                new_topics_words[key] = words
                new_topics_embeddings[key] = np.mean(embeddings, axis=0)

        return new_topics_words,new_topics_embeddings

    def update_BERTopic_model(self,mapped_topics):
        # Update documents and topics
        self.BERTopic_model.topic_mapper.add_mappings(mapped_topics)
        self.BERTopic_model._update_topic_size(self.df)
        self.BERTopic_model._extract_topics(self.df)
        self.BERTopic_model._update_topic_size(self.df)

    def new_model_topic_words(self,new_topics_words):
        sentence_model = SentenceTransformer("all-mpnet-base-v2")

        new_model_topics = {}

        if self.BERTopic_model is not None:
            for key,words in new_topics_words.items():
                topic_words = []
                for word in words:
                    embeddings = sentence_model.encode(word)
                    sim = cosine_similarity(self.BERTopic_model.topic_embeddings[key+1].reshape(1, -1),embeddings.reshape(1, -1))
                    topic_words.append((word,sim[0][0]))

                topic_words.sort(key = lambda x: x[1], reverse=True) 
                topic_words = list(dict.fromkeys(topic_words))
                new_model_topics[key]=topic_words
            self.BERTopic_model.topics = new_model_topics
        else:
            for key,words in new_topics_words.items():
                topic_words = []
                for word in words:
                    embeddings = sentence_model.encode(word)
                    sim = cosine_similarity(self.BERTopic_model.topic_embeddings[key+1].reshape(1, -1),embeddings.reshape(1, -1))
                    topic_words.append((word,sim[0][0]))

                topic_words.sort(key = lambda x: x[1], reverse=True) 
                topic_words = list(dict.fromkeys(topic_words))
                new_model_topics[key]=topic_words

        return new_model_topics

    def topic_names_dict(self,new_model_topics,num_words=5):
        nlp = spacy.load("en_core_web_sm")

        topics = {}
        topic_names = []

        for key,value in new_model_topics.items():
            topic_words = [i[0] for i in value]

            unallowed_postags=['X', 'SYM', 'PUNCT', 'NUM','SPACE','ADP','AUX','CONJ','CCONJ','DET','PRON','SCONJ']
            doc = nlp(' '.join(topic_words))
            words = [token.text for token in doc if token.pos_ not in unallowed_postags and token.lemma_!='-PRON-']
                
            if len(words)>=num_words:
                topics[str(key)] = " ".join(words[:num_words])
            else:
                topics[str(key)] = " ".join(topic_words[:num_words])
            topic_names.append(" ".join(topic_words[:num_words]))

        return topics

    def plot_dendrogram(self,level,new_rank,topics):

        from scipy.cluster.hierarchy import linkage
        import plotly.figure_factory as ff
        import plotly.figure_factory._dendrogram as original_dendrogram
        from plotly import exceptions, optional_imports

        def modified_dendrogram_traces(
            self, X, colorscale, distfun, linkagefun, hovertext, color_threshold
        ):
            """
            Calculates all the elements needed for plotting a dendrogram.

            :param (ndarray) X: Matrix of observations as array of arrays
            :param (list) colorscale: Color scale for dendrogram tree clusters
            :param (function) distfun: Function to compute the pairwise distance
                                    from the observations
            :param (function) linkagefun: Function to compute the linkage matrix
                                        from the pairwise distances
            :param (list) hovertext: List of hovertext for constituent traces of dendrogram
            :rtype (tuple): Contains all the traces in the following order:
                (a) trace_list: List of Plotly trace objects for dendrogram tree
                (b) icoord: All X points of the dendrogram tree as array of arrays
                    with length 4
                (c) dcoord: All Y points of the dendrogram tree as array of arrays
                    with length 4
                (d) ordered_labels: leaf labels in the order they are going to
                    appear on the plot
                (e) P['leaves']: left-to-right traversal of the leaves

            """
            np = optional_imports.get_module("numpy")
            scp = optional_imports.get_module("scipy")
            sch = optional_imports.get_module("scipy.cluster.hierarchy")
            scs = optional_imports.get_module("scipy.spatial")

            d = distfun(X)
            Z = linkagefun(d)
            P = sch.dendrogram(
                Z,
                orientation=self.orientation,
                labels=self.labels,
                no_plot=True,
                color_threshold=color_threshold,
                p = p,
                truncate_mode = 'lastp'
            )

            icoord = scp.array(P["icoord"])
            dcoord = scp.array(P["dcoord"])
            ordered_labels = scp.array(P["ivl"])
            color_list = scp.array(P["color_list"])
            colors = self.get_color_dict(colorscale)

            trace_list = []

            for i in range(len(icoord)):
                # xs and ys are arrays of 4 points that make up the '∩' shapes
                # of the dendrogram tree
                if self.orientation in ["top", "bottom"]:
                    xs = icoord[i]
                else:
                    xs = dcoord[i]

                if self.orientation in ["top", "bottom"]:
                    ys = dcoord[i]
                else:
                    ys = icoord[i]
                color_key = color_list[i]
                hovertext_label = None
                if hovertext:
                    hovertext_label = hovertext[i]
                trace = dict(
                    type="scatter",
                    x=np.multiply(self.sign[self.xaxis], xs),
                    y=np.multiply(self.sign[self.yaxis], ys),
                    mode="lines",
                    marker=dict(color=colors[color_key]),
                    text=hovertext_label,
                    hoverinfo="text",
                )

                try:
                    x_index = int(self.xaxis[-1])
                except ValueError:
                    x_index = ""

                try:
                    y_index = int(self.yaxis[-1])
                except ValueError:
                    y_index = ""

                trace["xaxis"] = "x" + x_index
                trace["yaxis"] = "y" + y_index

                trace_list.append(trace)

            return trace_list, icoord, dcoord, ordered_labels, P["leaves"]

        original_dendrogram._Dendrogram.get_dendrogram_traces = modified_dendrogram_traces

        X = self.distance_matrix
        orientation = "left"
        width = 1000
        height = 600
        fig = ff.create_dendrogram(self.distance_matrix,
                                orientation=orientation,
                                linkagefun=lambda x: linkage(x, "ward"),
                                color_threshold=level+1)

        # Create nicer labels
        # axis = "yaxis" if orientation == "left" else "xaxis"
        # new_labels = [[[str(topics[int(x)]), None]] + model.get_topic(topics[int(x)])
        #               for x in fig.layout[axis]["ticktext"]]
        # new_labels = ["_".join([label[0] for label in labels[:4]]) for labels in new_labels]
        # new_labels = [label if len(label) < 30 else label[:27] + "..." for label in new_labels]

        # new_labels = [model.topic_names[topic] for topic in new_rank]
        new_labels = [str(topic)+' '+topics[str(topic)] for topic in new_rank]

        # Stylize layout
        fig.update_layout(
            plot_bgcolor='#ECEFF1',
            template="plotly_white",
            title={
                'text': "<b>Hierarchical Clustering",
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(
                    size=22,
                    color="Black")
            },
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
        )

        # Stylize orientation
        if orientation == "left":
            fig.update_layout(height=200+(15*len(topics)),
                            width=width,
                            yaxis=dict(tickmode="array",
                                        ticktext=new_labels))
            
            # Fix empty space on the bottom of the graph
            y_max = max([trace['y'].max()+5 for trace in fig['data']])
            y_min = min([trace['y'].min()-5 for trace in fig['data']])
            fig.update_layout(yaxis=dict(range=[y_min, y_max]))

        else:
            fig.update_layout(width=200+(15*len(topics)),
                            height=height,
                            xaxis=dict(tickmode="array",
                                        ticktext=new_labels))
            
        fig.show()

        return fig
