import numpy as np


def code_complement(data):

    rows, cols = data.shape
    code_complemented = np.empty((0, cols))

    for row in data:
        code_complemented = np.vstack((code_complemented, row))
        code_complemented = np.vstack((code_complemented, 1 - row))

    return code_complemented


def create_network(cnt_features=None):

    weight = np.ones((cnt_features, 0))
    network = {'cnt_features': cnt_features,
               'cnt_categories': 0,
               'max_cnt_categories': 100,
               'weight': weight,
               'vigilance': 0.75,
               'bias': 0.000001,
               'cnt_epochs': 100,
               'learning_rate': 1}

    return network


def categorize(art_network=None, data=None):

    categorization = ones(1, numSamples)
    for sampleNumber in arange(1, numSamples).reshape(-1):
        currentData = data(arange(), sampleNumber)
        bias = art_network.bias
        categoryActivation = activate_categories(currentData, art_network.weight, bias)
        sortedActivations, sortedCategories = sort(- categoryActivation, nargout=2)
        resonance = 0
        match = 0
        numSortedCategories = length(sortedCategories)
        currentSortedIndex = 1
        while (logical_not(resonance)):

            currentCategory = sortedCategories(currentSortedIndex)
            currentWeightVector = art_network.weight(arange(), currentCategory)
            match = calculate_match(currentData, currentWeightVector)
            if (logical_or((match > art_network.vigilance), (match >= 1))):
                categorization[1, sampleNumber] = currentCategory
                resonance = 1
            else:
                if (currentSortedIndex == numSortedCategories):
                    categorization[1, sampleNumber] = - 1
                    resonance = 1
                else:
                    currentSortedIndex = currentSortedIndex + 1

    return categorization

def activate_categories(input_data, weight, bias):

    cnt_features, cnt_categories =  weight.shape
    category_activation = np.ones(1, cnt_categories)

    for j in range(cnt_categories):
        match_vector = min(input_data, weight[: ,j])
        weight_length = sum(weight[:, j])
        category_activation[1, j] = sum(match_vector) / (bias+weight_length)

    return category_activation


def add_new_category(weight):

    cnt_features, cnt_categories = np.shape(weight)
    new_category = np.ones(cnt_features, 1)
    resized_weight = weight.append([new_category])

    return resized_weight

def calculate_match(input_data, weight_vector):

    match_vector = min(input_data, weight_vector)
    input_length = sum(input_data)
    if input_length == 0:
        match = 0
    else:
        match = sum(match_vector)/ input_length

    return match

def update_weight():

    numFeatures, numCategories = size(weight, nargout=2)

    for i in arange(1, numFeatures).reshape(-1):
        if (input_(i) < weight(i, categoryNumber)):
            weight[i, categoryNumber] = (dot(learningRate, input_(i))) + (
                dot((1 - learningRate), weight(i, categoryNumber)))
            weightChange = 1
    updatedWeight = copy(weight)

    return updatedWeight, weightChange

def learn(network, data):

    new_art_network = cellarray([])
    categorization = ones(1, numSamples)

    for epochNumber in arange(1, art_network.numEpochs).reshape(-1):
        numChanges = 0

        for sampleNumber in arange(1, numSamples).reshape(-1):
            currentData = data(arange(), sampleNumber)
            bias = art_network.bias
            categoryActivation = ART_Activate_Categories(currentData, art_network.weight, bias)
            sortedActivations, sortedCategories = sort(- categoryActivation, nargout=2)
            resonance = 0
            match = 0
            numSortedCategories = length(sortedCategories)
            currentSortedIndex = 1
            while (logical_not(resonance)):

                if (numSortedCategories == 0):
                    resizedWeight = ART_Add_New_Category(art_network.weight)
                    resizedWeight, weightChange = ART_Update_Weights(currentData, resizedWeight, 1,
                                                                     art_network.learningRate, nargout=2)
                    art_network.weight = copy(resizedWeight)
                    art_network.numCategories = copy(art_network.numCategories + 1)
                    categorization[1, sampleNumber] = 1
                    numChanges = numChanges + 1
                    resonance = 1
                    break

                currentCategory = sortedCategories(currentSortedIndex)
                currentWeightVector = art_network.weight(arange(), currentCategory)
                match = ART_Calculate_Match(currentData, currentWeightVector)
                if (logical_or((match > art_network.vigilance), (match >= 1))):
                    art_network.weight, weightChange = ART_Update_Weights(currentData, art_network.weight,
                                                                          currentCategory, art_network.learningRate,
                                                                          nargout=2)
                    categorization[1, sampleNumber] = currentCategory

                    if (weightChange == 1):
                        numChanges = numChanges + 1
                    resonance = 1
                else:

                    if (currentSortedIndex == numSortedCategories):
                        if (currentSortedIndex == art_network.maxNumCategories):
                            categorization[1, sampleNumber] = - 1
                            resonance = 1
                        else:
                            resizedWeight = ART_Add_New_Category(art_network.weight)
                            resizedWeight, weightChange = ART_Update_Weights(currentData, resizedWeight,
                                                                             currentSortedIndex + 1,
                                                                             art_network.learningRate, nargout=2)
                            art_network.weight = copy(resizedWeight)
                            art_network.numCategories = copy(art_network.numCategories + 1)
                            categorization[1, sampleNumber] = currentSortedIndex + 1
                            numChanges = numChanges + 1
                            resonance = 1
                    else:
                        currentSortedIndex = currentSortedIndex + 1

        if (numChanges == 0):
            break

    return new_art_network, categorization



