import warnings

import gvgen
import numpy as np
import pandas as pd
from graphviz import Source
from anytree import Node as TreeNode, RenderTree


warnings.filterwarnings('ignore')

TRANSLATOR = dict.fromkeys(['Alt', 'Bar', 'Fri', 'Hun', 'Rain', 'Res', 'y_label'], {0: 'No', 1: 'Yes'})
TRANSLATOR.update({
    'Pat': {0: 'None', 1: 'Some', 2: 'Full'},
    'Price': {0: '$', 1: '$$', 2: '$$$'},
    'Type': {0: 'French', 1: 'Thai', 2: 'Burger', 3: 'Italian'},
    'Est': {0: '0-10', 1: '10-30', 2: '30-60', 3: '>60'}
})


class Node:
    def __init__(self, entropy):
        self.entropy = entropy


class InteriorNode(Node):
    def __init__(self, attr, entropy, gain, remainder):
        super().__init__(entropy)
        self.attribute = attr
        self.entropy = entropy
        self.gain = gain
        self.remainder = remainder
        self.childs = dict()

    def add_child(self, val, tree):
        self.childs[val] = tree

    def __str__(self):
        return str('<< ' + self.attribute + ' >>\n\n' + 'Entropy = ' + '%.4f' % self.entropy
                   + '\n' + 'Gain = ' + '%.4f' % self.gain
                   + '\n' + 'Remainder = ' + '%.4f' % self.remainder)


class LeafNode(Node):
    def __init__(self, entropy, label):
        super().__init__(entropy)
        self.label = label

    def __str__(self):
        return '<< ' + TRANSLATOR['y_label'][self.label] + ' >>\n\n' + 'Entropy = ' + '%.4f' % self.entropy


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = None):
        self.trained_tree = None
        self.max_depth = max_depth

    def calc_binary_entropy(self, p):
        if p == 0 or p == 1:
            return 0
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def calc_plurality_value(self, X, y_label):
        p = len(X[X[y_label] == 1])
        n = len(X[X[y_label] == 0])
        entropy = self.calc_binary_entropy(np.divide(p, p + n))
        mode = p if p >= n else n
        return mode, entropy

    def calc_remainder(self, X, feature, y_label):
        vals = X[feature].unique()
        p = len(X[X[y_label] == 1])
        n = len(X[X[y_label] == 0])
        rem = 0.0

        for k in vals:
            pk = len(X[(X[feature] == k) & (X[y_label] == 1)])
            nk = len(X[(X[feature] == k) & (X[y_label] == 0)])
            rem += np.divide(pk + nk, p + n) * self.calc_binary_entropy(np.divide(pk, pk + nk))

        return rem

    def calc_info_gain(self, X, feature, y_label):
        p = len(X[X[y_label] == 1])
        n = len(X[X[y_label] == 0])
        entropy = self.calc_binary_entropy(np.divide(p, p + n))
        remainder = self.calc_remainder(X, feature, y_label)
        return entropy - remainder, remainder

    def select_feature(self, X, features, y_label):
        selected_feature = None
        gain = -np.inf
        remainder = 0

        for feature in features:
            _gain, _remainder = self.calc_info_gain(X, feature, y_label)
            if _gain > gain:
                selected_feature = feature
                gain = _gain
                remainder = _remainder

        return selected_feature, gain, remainder

    @staticmethod
    def check_all_have_same_classification(x):
        a = x.to_numpy()
        return (a[0] == a).all()

    def create_decision_tree(self, X, features, y_label, parent_Xs, depth):
        if X.empty:
            entropy, label = self.calc_plurality_value(parent_Xs, y_label)
            return LeafNode(entropy, label)

        if DecisionTreeClassifier.check_all_have_same_classification(X[y_label]):
            return LeafNode(0, X[y_label].iloc[0])

        if not features or (self.max_depth and depth == self.max_depth):
            entropy, label = self.calc_plurality_value(X, y_label)
            return LeafNode(entropy, label)

        feature, gain, remainder = self.select_feature(X, features, y_label)
        root = InteriorNode(feature, gain + remainder, gain, remainder)
        vals = X[feature].unique()

        for val in vals:
            new_feats = features.copy()
            new_feats.remove(feature)
            subtree = self.create_decision_tree(X[X[feature] == val], new_feats, y_label, X, depth + 1)
            root.add_child(val, subtree)

        return root

    def fit(self, X, y_label):
        features = list(X.columns)
        features.remove(y_label)
        self.trained_tree = self.create_decision_tree(X, features, y_label, None, 0)

    @staticmethod
    def make_tree_graphviz(graph, tree):
        if isinstance(tree, LeafNode):
            return graph.newItem(tree.__str__())

        if isinstance(tree, InteriorNode):
            item = graph.newItem(tree.__str__())
            for key, val in sorted(tree.childs.items(), key=lambda x: 1 if isinstance(x[1], InteriorNode) else 0):
                child = DecisionTreeClassifier.make_tree_graphviz(graph, val)
                link = graph.newLink(item, child)
                graph.propertyAppend(link, "label", TRANSLATOR[tree.attribute][key])
                graph.propertyAppend(link, "color", "red")
            return item

    def show_graphical_decision_tree_result(self):
        g = gvgen.GvGen()
        DecisionTreeClassifier.make_tree_graphviz(g, self.trained_tree)

        with open("graphviz_output.txt", 'w') as f:
            g.dot(f)

        with open("graphviz_output.txt", 'r') as f:
            lines = f.readlines()[1:]

        _str = ''.join(lines)
        src = Source(_str)
        src.render('Decision Tree Result', view=True)

    @staticmethod
    def drawTree(root, new_root):
        if isinstance(root, InteriorNode):
            for val, child in sorted(root.childs.items(), key=lambda x: 1 if isinstance(x[1], InteriorNode) else 0):
                node = TreeNode('label: ' + TRANSLATOR[root.attribute][val], parent=new_root)
                DecisionTreeClassifier.drawTree(child, TreeNode(child.__str__().replace('\n', ' '), parent=node))

    def show_decision_tree(self):
        head_print_node = TreeNode(self.trained_tree.__str__().replace('\n', ' '))
        DecisionTreeClassifier.drawTree(self.trained_tree, head_print_node)
        with open('decision_tree_res.txt', 'w', encoding="utf-8") as f:
            x = True
            for pre, fill, node in RenderTree(head_print_node):
                if x:
                    f.write('%s%s' % (pre, node.name))
                    x = False
                else:
                    f.write('\n%s%s' % (pre, node.name))


if __name__ == '__main__':
    rest_df = pd.read_csv('restaurant.csv')
    dtc = DecisionTreeClassifier()
    dtc.fit(rest_df, 'Goal')
    dtc.show_decision_tree()
    dtc.show_graphical_decision_tree_result()
