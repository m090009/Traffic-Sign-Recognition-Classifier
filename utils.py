import winsound
import pyglet
import matplotlib.pyplot as plt
import random
import os
import math
import csv
import numpy as np
from collections import Counter
from matplotlib import gridspec


def read_csv(file_path):
    with open(file_path, mode='r') as infile:
        with open(file_path, mode='r') as infile:
            reader = csv.reader(infile)
            labels_mapping = dict((rows[0], rows[1]) for rows in reader)
        if 'ClassId' in labels_mapping.keys():
            del labels_mapping['ClassId']
        return labels_mapping


def draw_table(dict, title=''):
    plt.table(cellText=list(dict.items()),
              colLabels=['ClassIDs', 'Labels'],
              loc='bottom')
    plt.title(title)
    plt.show()


def beep():
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)


def show_label_counts_with_mapping(label_counter, mapping):
    fig, ax = plt.subplots(figsize=(10, 15))
    # Getting graph items ready
    labels = label_counter.keys() if not mapping else [
        mapping[str(k)] for k in list(label_counter.keys())]
    counts = label_counter.values()
    y_pos = np.arange(len(labels))
    error = np.random.rand(len(labels))

    ax.barh(y_pos, counts, xerr=error, align='center', color='lightblue')
    # Labels from top to bottom
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylabel('Signs')
    ax.set_xlabel('Count')
    ax.set_title('Traffic signs labels and their counts within the Training data')
    for i, v in enumerate(counts):
        ax.text(v + 3, i + .25, str(v), color='#02b3e4', fontweight='bold')

    # Printing the maximum and minimum labels and their counts from the training dataset
    most_common = label_counter.most_common()
    if len(most_common) > 0:
        max_count = most_common[0]
        min_count = most_common[-1]
        print("The most common Sign is: {} with {} occurences".format(
            mapping[str(max_count[0])], max_count[1]))
        print("The least common Sign is: {} with {} occurences".format(
            mapping[str(min_count[0])], min_count[1]))
        print('That makes the overall average occurences: {}'.format(int(np.mean(list(counts)))))
    # Showing the graph
    plt.show()


def play_sound():
    music = pyglet.resource.media("/assets/Training_Completed.mp3")
    music.play()
    pyglet.app.run()

    # Show a plot of accuracy over epochs


def plot_accuracy(
        title,
        accuracy,
        val_accuracy,
        directory='./'):
    """
    Plot loss and print stats of weights using an example neural network
    """
    colors = ['r', 'b', 'g', 'c', 'y', 'k']
    label_accs = []
    label_loss = []

    plt.plot(accuracy, colors[0], label="training_accuracy")
    plt.plot(val_accuracy, colors[1], label="validation_accuracy")
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(directory + "/" + title + '.png', bbox_inches='tight')
    plt.show()


def plot_accuracies(
        title,
        accuracies,
        labels,
        directory='./'):
    """
    Plot loss and print stats of weights using an example neural network
    """
    colors = ['r', 'lightsalmon', 'b', 'mediumslateblue', 'g',
              'mediumseagreen', 'c', 'cyan', 'y', 'yellow', 'k', 'gray', 'm']
    # accuracies = [[....], [....]]
    cnt = 0
    for i in range(len(accuracies)):
        for j in range(len(accuracies[i])):
            plt.plot(accuracies[i][j], colors[cnt], label=labels[i][j])
            cnt += 1
    # plt.add_axes([0.1, 0.1, 0.6, 0.75])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(directory + "/" + title + '.png',  bbox_extra_artists=(lgd,))
    plt.show()


def wrap_words(line, wrap=5):
    if len(line.split(' ')) >= wrap:
        line = line.split(' ')
        line.insert(wrap, "\n")
        return' '.join(line)
    else:
        return line


def show_images(images,
                images_titles=[],
                image_name='',
                cmap=None,
                save=False,
                horizontal=False,
                cols=4):
    SAVE_DIR = 'test_images_output/'
    directory = ''
    # if horizontal:
    cols = cols
    rows = int(math.ceil(len(images) / cols))
    # else:
    #     rows = 8
    #     cols = int(math.ceil(len(images) / rows))
    plt.figure(figsize=(15, 15))
    # Wrapping all titles to two lines if they have more than 4 words using the wrap_words method
    images_titles = list(map(wrap_words, images_titles))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # use gray scale color map if there is only one channel
        if len(image.shape) == 3 and image.shape[2] == 1:
            cmap = 'gray'
            image = image.squeeze()
        if len(image.shape) == 2:
            cmap = 'gray'

        plt.imshow(image, cmap=cmap)
        if len(images_titles) == len(images):
            plt.title(images_titles[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    if save:
        if os.path.isdir(SAVE_DIR):
            directory = SAVE_DIR
        image_name = str(datetime.datetime.now()).split('.')[0].replace(' ', '').replace(
            ':', '').replace('-', '') if image_name == '' else image_name
        plt.savefig(directory + image_name + '.png', bbox_inches='tight')
        # plt.tight_layout()
    plt.show()


def draw_label_counts(label_counter, mapping=None):
    fig, ax = plt.subplots(figsize=(10, 15))

    # Getting graph items ready
    labels = label_counter.keys()
    counts = label_counter.values()
    y_pos = np.arange(len(labels))
    error = np.random.rand(len(labels))

    ax.barh(y_pos, counts, xerr=error, align='center', color='lightblue')
    # Labels from top to bottom
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_ylabel('Signs')
    ax.set_xlabel('Count')
    ax.set_title('Traffic signs labels and their counts within the Training data')
    for i, v in enumerate(counts):
        ax.text(v + 3, i + .25, str(v), color='#02b3e4', fontweight='bold')

    # Printing the maximum and minimum labels and their counts from the training dataset
    most_common = label_counter.most_common()
    if len(most_common) > 0:
        max_count = most_common[0]
        min_count = most_common[-1]
        print("The Sign with the most occurences is: {}".format(max_count))
        print("The Sign with the least occurences is: {}".format(min_count))
        print('That make the average occurences: {}'.format(np.mean(list(counts))))
    # Showing the graph
    plt.show()


def get_data_count(X_data, y_data, mapping=None):
    # Create a counter to count the occurences of a sign (label)
    values = mapping.values() if mapping else list(set(y_data))
    data_counter = Counter(values)

    # We count each label occurence and store it in our label_counter
    for label in y_data:
        if mapping:
            data_counter[mapping[str(label)]] += 1
        else:
            data_counter[label] += 1
    return data_counter


def show_images_prediction(images,
                           images_predictions,
                           images_predictions_values,
                           images_titles,
                           image_name='',
                           save=False,
                           horizontal=False, cols=5):
    SAVE_DIR = 'test_images_output/'
    directory = ''
    cmap = None
    # if horizontal:

    cols = cols
    rows = int(math.ceil(len(images) / cols))
    # else:
    #     rows = 8
    #     cols = int(math.ceil(len(images) / rows))
    fig = plt.figure(figsize=(15, 15))
    # Wrapping all titles to two lines if they have more than 4 words using the wrap_words method
    images_titles = list(map(wrap_words, images_titles))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # use gray scale color map if there is only one channel
        if len(image.shape) == 3 and image.shape[2] == 1:
            cmap = 'gray'
            image = image.squeeze()
        if len(image.shape) == 2:
            cmap = 'gray'

        plt.imshow(image, cmap=cmap)
        color = 'green' if images_predictions[i] else 'red'
        # plt.text(0.95, 0.01, '',
        #          verticalalignment='bottom', horizontalalignment='right',
        #          color=color, fontsize=12)

        if len(images_titles) == len(images):
            title = images_titles[i]
            if not images_predictions[i]:
                title += ' \nAS\n {}'.format(wrap_words(images_predictions_values[i]))
            plt.title(title, color=color, fontsize=12)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    if save:
        if os.path.isdir(SAVE_DIR):
            directory = SAVE_DIR
        image_name = str(datetime.datetime.now()).split('.')[0].replace(' ', '').replace(
            ':', '').replace('-', '') if image_name == '' else image_name
        plt.savefig(directory + image_name + '.png', bbox_inches='tight')
    #
    fig.tight_layout()
    plt.show()


def show_image_topk(image,
                    labels,
                    values,
                    image_title,
                    image_prediction,
                    image_name='',
                    save=False,
                    horizontal=False):
    # SAVE_DIR = 'test_images_output/'
    # directory = ''
    cmap = None
    # # plt.figure(figsize=(15, 7))
    # plt.subplots(1, 2, figsize=(15, 7))
    # # plt.subplots_adjust(hspace=0.4)
    # if len(image.shape) == 3 and image.shape[2] == 1:
    #     cmap = 'gray'
    #     image = image.squeeze()
    # if len(image.shape) == 2:
    #     cmap = 'gray'
    color = 'green' if image_prediction else 'red'
    title = wrap_words(image_title)
    # ax1 = plt.subplot(211)
    # ax1.set_title(title, color=color, fontsize=12)
    # ax1.imshow(image, cmap=cmap)
    # # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    # plt.axis('off')
    #
    # ax = plt.subplot(212)
    # draw_bars(ax, labels, values, image_title)
    # # plt.show()
    # plt.tight_layout()
    fig = plt.figure(figsize=(15, 7))

    # show original image
    fig.add_subplot(221)
    plt.title(title, color=color, fontsize=12)
    plt.axis('off')
    plt.imshow(image, cmap=cmap)

    fig.add_subplot(222)
    draw_bars(labels, values, image_title)

    plt.show()


def draw_bars(labels, values, title):
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.set_xscale("log")
    y_pos = np.arange(len(labels))
    error = np.random.rand(len(labels))
    colors = ['limegreen' if label == title else 'orangered' for label in labels]
    labels = list(map(wrap_words, labels))
    plt.barh(y_pos, values, xerr=error, align='edge', color=colors, tick_label=labels)

    plt.title('{} image prediction percentages %'.format(title))

    plt.yticks(y_pos)
    # plt.yticklabels(labels)
    # plt.xscale('log', nonposx='clip')
    plt.ylabel('Signs')
    plt.xlabel('Percentage%')
    plt.axis('tight')
    plt.gca().invert_yaxis()
    for i, v in enumerate(values):
        plt.text(v + 3, i + .25, str(v), color=colors[i], fontweight='bold')
    # ax
