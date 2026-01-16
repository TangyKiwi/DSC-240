# Starter code for Winter 2025 DSC 240 MP1

def run_train_test(training_input, testing_input):
    """
    Implement the training and testing procedure here. You are permitted
    to use additional functions but DO NOT change this function definition.
    You are permitted to use the numpy library but you must write
    your own code for the linear classifier.

    Inputs:
        training_input: list form of the training file
            e.g. [[3, 5, 5, 5],[.3, .1, .4],[.3, .2, .1]...]
        testing_input: list form of the testing file

    Output:
        Dictionary of result values

        IMPORTANT: YOU MUST USE THE SAME DICTIONARY KEYS SPECIFIED

        Example:
            return {
                "tpr": #your_true_positive_rate,
                "fpr": #your_false_positive_rate,
                "error_rate": #your_error_rate,
                "accuracy": #your_accuracy,
                "precision": #your_precision
            }
    """

    # handle training input
    # D N1 N2 N3
    # f_1_1 f_1_2 ... f_1_D
    # ...
    # f_N1_1 f_N1_2 ... f_N1_D
    # ...
    # f_N2_1 f_N2_2 ... f_N2_D
    # ...
    # f_N3_1 f_N3_2 ... f_N3_D

    D_train = training_input[0][0]
    N1_train = training_input[0][1]
    N2_train = training_input[0][2]
    N3_train = training_input[0][3]

    A_train = training_input[1:1+N1_train]
    B_train = training_input[1+N1_train:1+N1_train+N2_train]
    C_train = training_input[1+N1_train+N2_train:1+N1_train+N2_train+N3_train]

    # handle testing input
    D_test = testing_input[0][0]
    N1_test = testing_input[0][1]
    N2_test = testing_input[0][2]
    N3_test = testing_input[0][3]

    A_test = testing_input[1:1+N1_test]
    B_test = testing_input[1+N1_test:1+N1_test+N2_test]
    C_test = testing_input[1+N1_test+N2_test:1+N1_test+N2_test+N3_test]

    # ==========================================================================
    # training
    # ==========================================================================

    # compute centroid of each class
    def compute_centroid(data):
        centroid = [0] * D_train
        for point in data:
            for i in range(D_train):
                centroid[i] += point[i]
        centroid = [x / len(data) for x in centroid]
        return centroid
    
    centroid_A = compute_centroid(A_train)
    centroid_B = compute_centroid(B_train)
    centroid_C = compute_centroid(C_train)

    # discriminant function between each pair of classes
    def discriminant_function(x, centroid1, centroid2):
        dot1 = sum(x[i] * centroid1[i] for i in range(D_train))
        dot2 = sum(x[i] * centroid2[i] for i in range(D_train))
        norm1 = sum(centroid1[i] ** 2 for i in range(D_train))
        norm2 = sum(centroid2[i] ** 2 for i in range(D_train))
        return dot1 - dot2 - 0.5 * (norm1 - norm2)
    
    # ==========================================================================
    # testing
    # ==========================================================================

    # for each instance in testing data, use discriminant function to classify
    # point, prioritizing A > B > C in case of ties
    # first use A/B
    # if classified as A, then decide A or C
    # if classified as B, then decide B or C
    # keep track of TP, TN, FP, FN

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for test_data, true_class in [(A_test, 'A'), (B_test, 'B'), (C_test, 'C')]:
        for x in test_data:
            g_AB = discriminant_function(x, centroid_A, centroid_B)
            if g_AB > 0:
                # classified as A or C
                g_AC = discriminant_function(x, centroid_A, centroid_C)
                if g_AC > 0:
                    predicted_class = 'A'
                else:
                    predicted_class = 'C'
            else:
                # classified as B or C
                g_BC = discriminant_function(x, centroid_B, centroid_C)
                if g_BC > 0:
                    predicted_class = 'B'
                else:
                    predicted_class = 'C'

            if true_class == 'A':
                if predicted_class == 'A':
                    TP += 1
                else:
                    FN += 1

    # compute metrics
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    error_rate = (FP + FN) / (TP + TN + FP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return {
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
        "accuracy": accuracy,
        "precision": precision
    }


    

    
