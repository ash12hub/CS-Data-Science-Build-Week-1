import numpy


class KNearestNeighbours:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train, k=1):
        self.k = k

        train = []
        for i in range(len(x_train)):
            train.append(x_train[i])
            train[i].append(y_train[i])

        self.train = train

    def predict(self, x_test):
        def euclidean_distance(self, instance1, instance2, length):
            distance = 0
            for i in range(length):
                distance += pow((instance1[i] - instance2[i]), 2)
            return numpy.sqrt(distance)

        length = len(x_test[0]) - 1
        predictions = []
        for j in range(len(x_test)):
            distances = []
            for i in range(len(self.train)):
                dist = self.euclidean_distance(self.train[i], x_test[j], length)
                distances.append([self.train[i], dist])
            distances.sort(key=lambda x: x[1])

            neighboors = []
            for i in range(self.k):
                neighboors.append(distances[i][0])
            self.neighboors = neighboors

            count = {}
            for i in range(len(neighboors)):
                prediction = neighboors[i][-1]
                if prediction in count:
                    count[prediction] += 1
                else:
                    count[prediction] = 1
            if len(count) > 1:
                sorted_count = sorted(count, key=lambda x: x[1], reverse=True)
                predictions.append(sorted_count[0])
            else:
                predictions.append(list(count.keys())[0])

            return predictions

    def get_accuracy(actual, predicted):
        count = 0
        correct = 0
        for i in range(len(actual)):
            count += 1
            if predicted[i] == actual[i]:
                correct += 1
        return correct/count
