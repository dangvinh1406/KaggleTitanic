import csv
import numpy

class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def process_data(filename):
        data = []
        with open(filename, "r") as file:
            reader = csv.reader(file, skipinitialspace=True)
            for record in reader:
                data.append(record)
        return data

    @staticmethod
    def convert_data(data):
        header = dict(zip(data[0], range(len(data[0]))))
        data = data[1:]
        X = []
        for record in data:
            vector = []

            try:
                vector.append(float(record[header["PassengerId"]]))
            except:
                pass

            try:
                vector.append(float(record[header["Survived"]]))
            except:
                pass

            pClass = [0]*3
            pClass[int(record[header["Pclass"]])-1] = 1
            vector += pClass

            sex = [0, 0]
            if record[header["Sex"]] == "male":
                sex[0] = 1
            elif record[header["Sex"]] == "female":
                sex[1] = 1
            vector += sex

            embarked = [0]*3
            if record[header["Embarked"]] == "C":
                embarked[0] = 1
            elif record[header["Embarked"]] == "Q":
                embarked[1] = 1
            elif record[header["Embarked"]] == "S":
                embarked[2] = 1
            vector += embarked

            vector += Preprocessor.process_string_features(
                record[header["Name"]], record[header["Cabin"]], record[header["Ticket"]]) # 16 parameters
            try:
                vector.append(float(record[header["Fare"]]))
            except:
                vector.append(0)

            try:
                vector.append(float(record[header["Age"]]))
            except:
                vector.append(0)

            try:
                vector.append(float(record[header["SibSp"]]))
            except:
                vector.append(0)

            try:
                vector.append(float(record[header["Parch"]]))
            except:
                vector.append(0)

            vector = numpy.array(vector, dtype=numpy.float32)
            X.append(vector)
        return X

    @staticmethod
    def scale_linear_by_column(rawpoints, maxs=None, mins=None, high=1.0, low=0.0):
        if maxs is None or mins is None:
            mins = numpy.min(rawpoints, axis=0)
            maxs = numpy.max(rawpoints, axis=0)
        rng = maxs-mins
        return high-(((high-low)*(maxs-rawpoints))/rng), maxs, mins

    @staticmethod
    def process_string_features(name, cabins, ticket):
        result = []
        if "Mr." in name:
            result.append(1)
        else:
            result.append(0)

        if "Miss." in name:
            result.append(1)
        else:
            result.append(0)

        if "Mrs." in name:
            result.append(1)
        else:
            result.append(0)

        if "Master." in name:
            result.append(1)
        else:
            result.append(0)

        if "Col." in name:
            result.append(1)
        else:
            result.append(0)

        if "Mlle." in name:
            result.append(1)
        else:
            result.append(0)

        if "Capt." in name:
            result.append(1)
        else:
            result.append(0)

        if "Dr." in name:
            result.append(1)
        else:
            result.append(0)

        if "." in ticket:
            result.append(1)
        else:
            result.append(0)

        if "/" in ticket:
            result.append(1)
        else:
            result.append(0)

        try:
            number = int(ticket)
            result.append(1)
        except:
            result.append(0)

        # define 5 cabins
        cabinFeature = [0]*5
        cabins = cabins.split(" ")
        i = 0
        for cabin in cabins:
            if cabin:
                cabinFeature[i] = int(str(ord(cabin[0]))+cabin[1:])
                i += 1

        result.append(len(ticket))

        return result