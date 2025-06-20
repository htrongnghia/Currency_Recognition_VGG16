import pickle
import numpy as np
import matplotlib.pyplot as plt

def plot_model_history(history_file, acc='accuracy', val_acc='val_accuracy'):
    # Load history from the pickle file
    with open(history_file, 'rb') as f:
        model_history = pickle.load(f)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    axs[0].plot(range(1, len(model_history[acc]) + 1), model_history[acc])
    axs[0].plot(range(1, len(model_history[val_acc]) + 1), model_history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history[acc]) + 1, step=max(1, len(model_history[acc]) // 10)))
    axs[0].legend(['train', 'val'], loc='best')

    # Plot loss
    axs[1].plot(range(1, len(model_history['loss']) + 1), model_history['loss'])
    axs[1].plot(range(1, len(model_history['val_loss']) + 1), model_history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history['loss']) + 1, step=max(1, len(model_history['loss']) // 10)))
    axs[1].legend(['train', 'val'], loc='best')

    plt.savefig('training_history_plot.png')
    plt.show()

# Example usage
plot_model_history('training_history.pkl')
