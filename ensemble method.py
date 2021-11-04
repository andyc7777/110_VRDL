"""# Ensemble Method Attemption
##### Code ref. from https://ensemble-pytorch.readthedocs.io/en/latest/
"""

# Set the Logger
logger = set_logger('classification')

# Define the ensemble
ens_model = VotingClassifier(
    estimator=resnet50,
    n_estimators=10,
    cuda=True,
)

# Set the optimizer
ens_model.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

# Set the learning rate scheduler
ens_model.set_scheduler(
    "CosineAnnealingLR",                    # type of learning rate scheduler
    T_max=10,                           # additional arguments on the scheduler
)

# Train and Evaluate
ens_model.fit(
    training_data_loader,
    epochs=5,
    test_loader=val_data_loader,
)

"""# Load saved Ensemble Model"""

ens_load_model = VotingClassifier(
    estimator=resnet50,
    n_estimators=10,
    cuda=True,
)
load(ens_load_model, './')  # reload

"""# Inference of Ensemble Model
##### Code ref. from Homework of NTU Prof.Hung-Yi Lee's lecture.
"""

# Evaluate the ensemble
device = "cuda" if torch.cuda.is_available() else "cpu"

predictions = []

# Iterate the testing set by batches.
for batch in tqdm(testing_data_loader):
    # print(batch)
    imgs = batch
    # We don't need gradient in testing,
    # and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        pred = ens_load_model.predict(imgs)

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(pred.argmax(dim=-1).cpu().numpy().tolist())

"""# Save Prediction
##### Modify the code from the TA.
##### And some code ref. from https://www.itread01.com/question/aXgwZQ==.html
"""

for i in range(len(predictions)):
    if predictions[i] < 100 and predictions[i] >= 10:
        predictions[i] = '0' + str(predictions[i])
    elif predictions[i] < 10:
        predictions[i] = '00' + str(predictions[i])

fh = open('testing_img_order.txt', 'r')
test_images = []
for line in fh:
    line = line.strip('\n')
    line = line.rstrip()
    words = line.split()
    test_images.append((str(words)[2:10]))


# ref:https://www.itread01.com/question/aXgwZQ==.html
def _finditem(obj, key):
    if key in obj.keys():
        print('有')
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            item = _finditem(v, key)
            if item is not None:
                return item


def find(dic, input_key):
    if str(input_key) in dic.keys():
        return dic[str(input_key)]

pred_class = []
for i in range(len(predictions)):
    class_name = find(class_dict, predictions[i])
    pred = str(predictions[i]) + '.' + str(class_name)
    pred_class.append(pred)

submission = {'file name': test_images, 'pred': pred_class}
submission = pd.DataFrame(submission)
submission.to_csv('ens_answer.txt', sep=' ', header=None, index=False)
