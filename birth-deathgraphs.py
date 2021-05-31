import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm


# config plot params
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# read in data from csv
df = pd.read_csv("datasetrate12.csv")

# create date time variable
date_time = pd.to_datetime(df.pop('DATE'), format='%m/%d/%y')


def get_NormArray(df, n, mode='total', linear=False):
    temp = []

    for i in range(len(df))[::-1]:

        if i >= n:  # there will be a traveling norm until we reach the initian n values.
            # those values will be normalized using the last computed values of F50,F75 and F25
            F50 = df[i - n:i].quantile(0.5)
            F75 = df[i - n:i].quantile(0.75)
            F25 = df[i - n:i].quantile(0.25)

        if linear == True and mode == 'total':
            v = 0.5 * ((df.iloc[i] - F50) / (F75 - F25)) - 0.5
        elif linear == True and mode == 'scale':
            v = 0.25 * df.iloc[i] / (F75 - F25) - 0.5
        elif linear == False and mode == 'scale':
            v = 0.5 * norm.cdf(0.25 * df.iloc[i] / (F75 - F25)) - 0.5

        else:  # even if strange values are given, it will perform full normalization with compression as default
            v = norm.cdf(0.5 * (df.iloc[i] - F50) / (F75 - F25)) - 0.5

        temp.append(v[0])
    return pd.DataFrame(temp[::-1])

# df2 = pd.read_csv("dataset.csv")
#
#
# #plot data here
# plot_cols = ['births']
# plot_features = df2[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)
#
# plt.show()

scaler = MinMaxScaler(feature_range=(0, 1))

### 1 - 0 SCALING ######
df_min = df.min()
df_max = df.max()

df = (df - df_min) / (df_max - df_min)


# for col in df:
#     df[col] = get_NormArray(pd.DataFrame(df[col]),12)
#
# # df = get_NormArray(df, 10)
#
# df.to_csv("datasetrate12.csv", index=False)

# plot_cols = ['births']
# plot_features = df[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)
#
# plt.show()

# get info about each feature (mean, median, etc)
# df.describe().transpose().to_csv(path_or_buf="/Users/joeyedell/Desktop/seminarproject/describe.csv")


# split data into training, validation, and test sets
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n * 0.70)]
val_df = df[int(n * 0.70):int(n * 1)]
test_df = df[int(n * .96):]

num_features = df.shape[1]

# normalize features: Standardization (Z-score Normalization)
train_mean = train_df.mean()
train_std = train_df.std()
val_mean = val_df.mean()
val_std = val_df.std()
test_mean = test_df.mean()
test_std = test_df.std()

train_min = train_df.min()
train_max = train_df.max()
val_min = val_df.min()
val_max = val_df.max()
test_min = test_df.min()
test_max = test_df.max()


# train_df = (train_df - train_min) / (train_max - train_min)
# val_df = (val_df - val_min) / (val_max - val_min)


# train_df = (train_df - train_mean) / train_std
# val_df = (val_df - val_mean) / val_std
# test_df = (test_df - test_mean) / test_std

# plot_cols = ['births']
# plot_features = df[plot_cols]
# plot_features.index = date_time
# _ = plot_features.plot(subplots=True)
#
# plt.show()

#plot normalized features
# #why im doing this: https://en.wikipedia.org/wiki/Feature_scaling
# df_std = (df - train_mean) / train_std
# df_std = df_std.melt(var_name='Column', value_name='Normalized')
# plt.figure(figsize=(12, 6))
# ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
# _ = ax.set_xticklabels(df.keys(), rotation=90)


# plt.show()


# data windowing
class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


WindowGenerator.split_window = split_window
###########################################
###########################################
###########################################
CONV_WIDTH = 3
LABEL_WIDTH = 12
INPUT_WIDTH = 10
OUTSTEPS = 12
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
w2 = WindowGenerator(
    input_width=OUTSTEPS    ,
    label_width=OUTSTEPS,
    shift=OUTSTEPS,
    label_columns=['births'])
###########################################
###########################################
###########################################

print("size")
print(train_df.size)

# Stack three slices, the length of the total window:
example_window = tf.stack([np.array(df[85:85+w2.total_window_size]),
                           #np.array(df[:w2.total_window_size]),
                           #np.array(df[:w2.total_window_size])
                           ])

example_inputs, example_labels = w2.split_window(example_window)

print(example_inputs)
print(example_labels)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'labels shape: {example_labels.shape}')

w2.example = example_inputs, example_labels


def plot(self, model=None, plot_col='births', max_subplots=1):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        #plt.subplot(max_n, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(range(109), df['births'],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue
        print(range(96,108))
        print(self.label_indices)

        plt.scatter(range(97,109), labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.plot(range(97,109), predictions[n, :, label_col_index],
                        marker='X', #edgecolors='k',
                        label='Predictions',
                        c='#ff7f0e',
                        #s=64
                        )


        if n == 0:
            plt.legend()

    plt.xlabel('Time')



WindowGenerator.plot = plot


# w2.plot()


# w2.plot(plot_col='births')

# plt.show()

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32, )

    ds = ds.map(self.split_window)

    return ds


WindowGenerator.make_dataset = make_dataset


@property
def train(self):
    return self.make_dataset(self.train_df)


@property
def val(self):
    return self.make_dataset(self.val_df)


# @property
# def test(self):
#     return self.make_dataset(self.test_df)


@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        print("here")
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result


WindowGenerator.train = train
WindowGenerator.val = val
# WindowGenerator.test = test
#WindowGenerator.example = example


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.SimpleRNNCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)


def warmup(self, inputs):
    # inputs.shape => (batch, time, features)
    # x.shape => (batch, lstm_units)
    x, *state = self.lstm_rnn(inputs)

    # predictions.shape => (batch, features)
    prediction = self.dense(x)
    return prediction, state


def call(self, inputs, training=None):
    # Use a TensorArray to capture dynamically unrolled outputs.
    predictions = []
    # Initialize the lstm state
    prediction, state = self.warmup(inputs)

    # Insert the first prediction
    predictions.append(prediction)

    # Run the rest of the prediction steps
    for n in range(1, self.out_steps):
        # Use the last prediction as input.
        x = prediction
        # Execute one lstm step.
        x, state = self.lstm_cell(x, states=state,
                                  training=training)
        # Convert the lstm output to a prediction.
        prediction = self.dense(x)
        # Add the prediction to the output
        predictions.append(prediction)

    # predictions.shape => (time, batch, features)
    predictions = tf.stack(predictions)
    # predictions.shape => (batch, time, features)
    predictions = tf.transpose(predictions, [1, 0, 2])
    return predictions


FeedBack.call = call

FeedBack.warmup = warmup

feedback_model = FeedBack(units=32, out_steps=OUTSTEPS)

val_performance = {}
performance = {}

multi_val_performance = {}
multi_performance = {}

###########
MAX_EPOCHS = 50


###########
def compile_and_fit(model, window, patience=2):
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                 patience=patience,
    #                                               mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        # callbacks=[early_stopping]
                        )
    return history

# linear = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1)
# ])
# history = compile_and_fit(linear, w2)
#
# val_performance['Linear'] = linear.evaluate(w2.val)
# #performance['Linear'] = linear.evaluate(w2.test, verbose=0)
# w2.plot(linear)
# plt.show()

# dense = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dropout(rate=.2),
#     tf.keras.layers.Dense(units=64, activation='sigmoid'),
#     tf.keras.layers.Dense(units=1)
# ])
# history = compile_and_fit(dense, w2)
#
# val_performance['Dense'] = dense.evaluate(w2.val)
# #performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
# w2.plot(dense)
# plt.show()

# conv_model = tf.keras.Sequential([
#     tf.keras.layers.Conv1D(filters=64,
#                            kernel_size=(CONV_WIDTH,),
#                            activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1),
# ])
#
# history = compile_and_fit(conv_model, w2)
#
# IPython.display.clear_output()
# val_performance['Conv'] = conv_model.evaluate(w2.val)
# # performance['Conv'] = conv_model.evaluate(w2.test)
#
# w2.plot(conv_model)
# plt.show()
#
# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.RNN(tf.keras.layers.SimpleRNNCell(64), return_sequences=True), #GOOD
#     # #tf.keras.layers.ActivityRegularization(),
#     # tf.keras.layers.RNN(tf.keras.layers.LSTMCell(64), return_sequences=True),
#
#     # tf.keras.layers.LSTM(64, return_sequences=True),
#     tf.keras.layers.Dropout(rate=.2), #GOOD
#     tf.keras.layers.Dense(units=32, activation='sigmoid'), #GOOD
#
#     #tf.keras.layers.LSTM(16, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=1) #GOOD
# ])
#
# history = compile_and_fit(lstm_model, w2)
#
# IPython.display.clear_output()
# val_performance['LSTM'] = lstm_model.evaluate(w2.val)
# #performance['LSTM'] = lstm_model.evaluate(w2.test)
# w2.plot(lstm_model)
# plt.show()

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(64, return_sequences=False),

    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUTSTEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUTSTEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, w2)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(w2.val)
#multi_performance['LSTM'] = multi_lstm_model.evaluate(w2.test, verbose=0)
w2.plot(multi_lstm_model)
plt.show()


# CONV_WIDTH = 3
# multi_conv_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
#     # Shape => [batch, 1, conv_units]
#     tf.keras.layers.Conv1D(64, activation='tanh', kernel_size=(CONV_WIDTH)),
#     # Shape => [batch, 1,  out_steps*features]
#     tf.keras.layers.Dense(OUTSTEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros()),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUTSTEPS, num_features])
# ])
#
# history = compile_and_fit(multi_conv_model, w2)
#
# IPython.display.clear_output()
#
# multi_val_performance['Conv'] = multi_conv_model.evaluate(w2.val)
# # multi_performance['Conv'] = multi_conv_model.evaluate(w2.test, verbose=0)
# w2.plot(multi_conv_model)
# plt.show()


# history = compile_and_fit(feedback_model, w2)
#
# IPython.display.clear_output()
#
# multi_val_performance['AR LSTM'] = feedback_model.evaluate(w2.val)
# #multi_performance['AR LSTM'] = feedback_model.evaluate(w2.test)
#
# # # plot data here
# # plot_features = df['births']
# # plot_features.index = date_time
# # _ = plot_features.plot(subplots=True)
# #
# # #plt.show()
#
# w2.plot(feedback_model)
# plt.show()

# plt.bar(x = range(len(train_df.columns)),
#        height=lstm_model.layers[0].kernel[:,0].numpy())
# axis = plt.gca()
# axis.set_xticks(range(len(train_df.columns)))
# _ = axis.set_xticklabels(train_df.columns, rotation=90)
# plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

plt.show()

# training perdio start as soon as i can and end after great recession late 2010
# see how it does 1 quarter ahead
# do not randomly assign biases

# index


#Add borodo-duca model
#test on data spread,
#instiute for supply management
#try to improve duca model
#german employment/manufacturying
#EPU series on FRED
#willingness to lend
#credit spreads
#baker bloom uncertinity

#tax changes effect on business

#IMPGSC
