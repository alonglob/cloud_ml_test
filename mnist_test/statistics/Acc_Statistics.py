import json
import matplotlib.pyplot as plt

# FILES MUST BE IN SAME FOLDER FOR THE LOVE OF CHRIST

def main():
    # displays the test values from 90%+ validation
    # shows precise final Validation Accuracy
    plt.figure(1)
    plot_full('model1.json')
    plot_full('model2.json')
    plt.axis([500, 10000, 0, 0.05])
    plt.legend(['model1', 'model2'])

    plt.xlabel('Epochs')
    plt.ylabel('Validation Percentage')

    # displays the train values before 95%
    # shows which computation converged faster to the 95%
    plt.figure(2)
    plot_full('model1.json')
    plot_full('model2.json')
    plt.legend(['model1', 'model2'])
    plt.axis([0, 10000, 0, 1])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Percentage')

    plt.show()

def plot_full(fileName):
    data = reader(fileName)

    epoch, acc = split(data)

    plt.plot(epoch, acc, label=fileName)


def plot_90(fileName):
    data = reader(fileName)
    data = data[3:]

    epoch, acc = split(data)

    plt.plot(epoch, acc, label=fileName)


def reader(name):
    """ name - string of the data json file, eg: 'data.json
    reads the file 'name' and cuts its first value in each index.
    outputting [epoch, Acc]"""

    try:
        with open(name) as data_file:
            data = json.load(data_file)

            data_new = []
            for x, y, z in data:
                data_new.append([y, z])

            return data_new
    except ValueError:
        print("seems like opening the file causes an error, please check file name!")
    pass


def split(data):
    epochList = []
    accList = []
    for x, y in data:
        epochList.append(x)
        accList.append(y)

    return epochList, accList


if __name__ == '__main__':
    main()
