from sklearn import model_selection

from .constants import Constants

# Split training set into folds for Monte Carlo cross validation.
# According to research, the optimal number of split is around 2n,
# i.e. 200.
mc_cv = model_selection.ShuffleSplit(
    n_splits=Constants.SPLITS.value,
    test_size=Constants.TEST_SIZE.value,
    train_size=Constants.TRAIN_SIZE.value,
    random_state=Constants.RANDOM_STATE.value,
)
