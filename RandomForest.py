import csv
import math
from collections import Counter
import random
import time

class RandomForest():
    sub_decision_tree = []
    feature_size = 19
    sampling_percentage = 0.8
    nb_sub_decision_tree = 20

    def learn(self, training_set):
        for count in range(self.nb_sub_decision_tree):
            attr_list = random.sample(range(0, 56), self.feature_size)
            attr_list.append(57)
            counter = 0
            mapping = {}
            for attr in attr_list:
                mapping[counter] = attr
                counter = counter + 1
            samples_seq_list = random.sample(range(0, len(training_set)), int(len(training_set) * self.sampling_percentage))
            sampled_training_set = []
            for samples_seq in samples_seq_list:
                row = []
                for attr in attr_list:
                    row.append(training_set[samples_seq][attr])
                sampled_training_set.append(row)

            decision_tree = DecisionTree()
            decision_tree.classification_attr = len(attr_list) - 1
            decision_tree.attr_mapping = mapping
            decision_tree.learn(sampled_training_set)
            self.sub_decision_tree.append(decision_tree)

    def classify(self, test_instance):
        count = []
        for decision_tree in self.sub_decision_tree:
            updated_test_instance = []
            for i in decision_tree.attr_mapping.values()[:-1]:
                updated_test_instance.append(test_instance[i])
            result = decision_tree.classify(updated_test_instance)
            count.append(int(result))

        if sum(count) > len(count)/2 :
            return '1'
        else:
            return '0'




# Implement your decision tree below
class DecisionTree():
    tree = {}
    mean_benchmark_limit = []
    classification_attr = 57
    attr_mapping = {}

    def learn(self, training_set):
        # implement this function
        self.tree = {}
        col_count = len(training_set[0]) - 1

        attr_list = range(col_count)
        training_set_binary = self.change_binary(col_count, training_set)
        self.tree = self.decisionTreeImpl(training_set_binary, attr_list)

    def change_binary(self, col_count, training_set):
        total = [0.0] * col_count
        for x in training_set:
            for y in range(col_count):
                total[y] = total[y] + float(x[y])

        mean = [0.0] * col_count
        for y in range(col_count):
            mean[y] = total[y] / len(training_set)

        self.mean_benchmark_limit = mean

        training_set_binary = []
        for x in training_set:
            row = []
            for y in range(col_count):
                if float(x[y]) >= mean[y]:
                    row.append(1)
                else:
                    row.append(0)
            training_set_binary.append(row)

        counter = -1
        for x in training_set:
            counter = counter + 1
            training_set_binary[counter].append(x[self.classification_attr])
        return training_set_binary

    def find_higher_occurance(self, training_set_binary):
        count_0 = 0
        count_1 = 0
        for x in training_set_binary:
            if x[self.classification_attr] == '0':
                count_0 += 1
            else:
                count_1 += 1
        if count_0 >= count_1:
            return 0
        else:
            return 1

    def decisionTreeImpl(self, training_set_binary, attr_list):
        classification_list = []
        for x in training_set_binary:
            classification_list.append(x[self.classification_attr])

        if len(Counter(classification_list)) == 1:
            return int(classification_list[0])

        if len(attr_list) == 0:
            return self.find_higher_occurance(training_set_binary)

        element_list = {}
        rootNodeEntropy = self.root_entropy(training_set_binary)
        for x in attr_list:
            info_gain = rootNodeEntropy - self.entropy(training_set_binary, x)
            element_list[x] = info_gain

        max_info_gain = 0
        maxKey = attr_list[0]
        for key in element_list:
            if element_list[key] > max_info_gain:
                max_info_gain = element_list[key]
                maxKey = key

        #print "key " + str(maxKey) + " info gain " + str(max_info_gain)

        attr_list.remove(maxKey)
        #split_attr = "attr" + str(self.attr_mapping[maxKey])
        split_attr = "attr" + str(maxKey)
        sub_tree = {split_attr: []}
        tree_0 = []
        tree_1 = []

        for x in training_set_binary:
            if x[maxKey] == 0:
                tree_0.append(x)
            else:
                tree_1.append(x)
        ans_0 = self.decisionTreeImpl(tree_0, attr_list[:])
        ans_1 = self.decisionTreeImpl(tree_1, attr_list[:])

        sub_tree[split_attr].append(ans_0)
        sub_tree[split_attr].append(ans_1)

        return sub_tree

    def root_entropy(self, training_set_binary):
        if len(training_set_binary) == 0:
            return 0
        zero_count = 0
        one_count = 0
        for x in training_set_binary:
            if x[self.classification_attr] == '0':
                zero_count = zero_count + 1
            else:
                one_count = one_count + 1
        p0 = float(zero_count) / len(training_set_binary)
        p1 = float(one_count) / len(training_set_binary)
        ent = -p0 * math.log(p0, 2) + -p1 * math.log(p1, 2)
        return ent

    def entropy(self, training_set_binary, col_index):
        if len(training_set_binary) == 0:
            return 0
        zero_count = 0
        cnt_0_0 = 0
        cnt_0_1 = 0
        cnt_1_0 = 0
        cnt_1_1 = 0
        one_count = 0
        for x in training_set_binary:
            if x[col_index] == 0:
                zero_count = zero_count + 1
                if x[self.classification_attr] == '0':
                    cnt_0_0 = cnt_0_0 + 1
                else:
                    cnt_0_1 = cnt_0_1 + 1
            else:
                one_count = one_count + 1
                if x[self.classification_attr] == '0':
                    cnt_1_0 = cnt_1_0 + 1
                else:
                    cnt_1_1 = cnt_1_1 + 1

        if (cnt_0_0 + cnt_0_1) != 0:
            p0_0 = float(cnt_0_0) / (cnt_0_0 + cnt_0_1)
            p0_1 = float(cnt_0_1) / (cnt_0_0 + cnt_0_1)
        else:
            p0_0 = 0.0
            p0_1 = 0.0

        if (cnt_1_0 + cnt_1_1) != 0:
            p1_0 = float(cnt_1_0) / (cnt_1_0 + cnt_1_1)
            p1_1 = float(cnt_1_1) / (cnt_1_0 + cnt_1_1)
        else:
            p1_0 = 0.0
            p1_1 = 0.0

        if p0_0 != 0:
            e0_0 = -p0_0 * math.log(p0_0, 2)
        else:
            e0_0 = 0
        if p0_1 != 0:
            e0_1 = -p0_1 * math.log(p0_1, 2)
        else:
            e0_1 = 0
        if p1_0 != 0:
            e1_0 = -p1_0 * math.log(p1_0, 2)
        else:
            e1_0 = 0
        if p1_1 != 0:
            e1_1 = -p1_1 * math.log(p1_1, 2)
        else:
            e1_1 = 0

        ent = float(zero_count) / len(training_set_binary) * (e0_0 + e0_1) + float(one_count) / len(
            training_set_binary) * (e1_0 + e1_1)
        return ent

    # implement this function
    def classify(self, test_instance):
        #result = '0'  # baseline: always classifies as 0

        myTree = self.tree
        binary_test_instance = []
        for x in range(len(test_instance)):
            if float(test_instance[x]) >= self.mean_benchmark_limit[x]:
                binary_test_instance.append(1)
            else:
                binary_test_instance.append(0)
        result = self.classifyImpl(binary_test_instance, myTree)

        return str(result)

    def classifyImpl(self, binary_test_instance, subtree):
        for key in subtree:
            attr = int(key.strip("attr"))
            if binary_test_instance[attr] == 0:
                ans = subtree[key][0]
            else:
                ans = subtree[key][1]

            if not isinstance(ans, dict):
                return ans
            else:
                recursive_tree = ans
                return self.classifyImpl(binary_test_instance,recursive_tree)

def run_random_forest():
    # Load data set
    with open("spam.data.txt") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=" ")]
    print "Number of records: %d" % len(data)

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [x for i, x in enumerate(data) if (i % K != 3) and (i % K != 6) and (i % K != 9)]
    test_set = [x for i, x in enumerate(data) if (i % K == 3) or (i % K == 6) or (i % K == 9)]

    forest = RandomForest()
    forest.learn(training_set)
    results = []
    for instance in test_set:
        result = forest.classify(instance[:-1])
        results.append(result == instance[-1])

        # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print "accuracy: %.4f" % accuracy

def run_random_forest_sensetivity():
    # Load data set
    with open("spam.data.txt") as f:
        next(f, None)
        data = [tuple(line) for line in csv.reader(f, delimiter=" ")]
    print "Number of records: %d" % len(data)

    # Split training/test sets
    # You need to modify the following code for cross validation.
    K = 10
    training_set = [x for i, x in enumerate(data) if (i % K != 3) and (i % K != 6) and (i % K != 9)]
    test_set = [x for i, x in enumerate(data) if (i % K == 3) or (i % K == 6) or (i % K == 9)]

    for i in range(1,20):
        forest = RandomForest()
        forest.feature_size = i
        forest.learn(training_set)
        results = []
        for instance in test_set:
            result = forest.classify(instance[:-1])
            results.append(result == instance[-1])

            # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        print str(i) +" "+ str(accuracy)



if __name__ == "__main__":
    start_time = time.time()
    run_random_forest()
    print("--- %s seconds ---" % (time.time() - start_time))
