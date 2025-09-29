# src/main.py
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Import custom utility functions
from utils import load_and_preprocess_nsl_kdd, create_baseline_model

# Ensure TensorFlow uses the CPU to avoid potential GPU memory issues
tf.config.set_visible_devices([], 'GPU')

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_nsl_kdd()
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))

    # --- Train both models first ---
    print("\n--- Training Baseline Model ---")
    baseline_model = create_baseline_model(input_shape, num_classes)
    baseline_model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1)

    print("\n--- Training Defended Model (with Adversarial Training) ---")
    art_classifier_for_training = TensorFlowV2Classifier(model=baseline_model, nb_classes=num_classes, input_shape=input_shape, loss_object=SparseCategoricalCrossentropy(), clip_values=(0, 1))
    pgd_attack_for_training = ProjectedGradientDescent(estimator=art_classifier_for_training, eps=0.1, max_iter=10)
    X_train_adv = pgd_attack_for_training.generate(x=X_train[:20000])
    X_train_augmented = np.concatenate([X_train, X_train_adv], axis=0)
    y_train_augmented = np.concatenate([y_train, y_train[:20000]], axis=0)
    
    defended_model = create_baseline_model(input_shape, num_classes)
    defended_model.fit(X_train_augmented, y_train_augmented, epochs=10, batch_size=256, verbose=1)

    # --- Robustness Curve Analysis ---
    print("\n" + "="*50)
    print("Robustness Curve Analysis vs. Attack Strength")
    print("="*50)

    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    baseline_accuracies = []
    defended_accuracies = []

    art_classifier_baseline = TensorFlowV2Classifier(model=baseline_model, nb_classes=num_classes, input_shape=input_shape, loss_object=SparseCategoricalCrossentropy(), clip_values=(0, 1))
    art_classifier_defended = TensorFlowV2Classifier(model=defended_model, nb_classes=num_classes, input_shape=input_shape, loss_object=SparseCategoricalCrossentropy(), clip_values=(0, 1))

    for eps in epsilons:
        print(f"\n--- Testing with Epsilon = {eps:.2f} ---")
        if eps == 0.0:
            y_pred_baseline = np.argmax(baseline_model.predict(X_test, verbose=0), axis=1)
            y_pred_defended = np.argmax(defended_model.predict(X_test, verbose=0), axis=1)
        else:
            pgd_attack_baseline = ProjectedGradientDescent(estimator=art_classifier_baseline, eps=eps, max_iter=10)
            pgd_attack_defended = ProjectedGradientDescent(estimator=art_classifier_defended, eps=eps, max_iter=10)
            print("Generating adversarial samples for both models...")
            x_test_adv_baseline = pgd_attack_baseline.generate(x=X_test)
            x_test_adv_defended = pgd_attack_defended.generate(x=X_test)
            y_pred_baseline = np.argmax(baseline_model.predict(x_test_adv_baseline, verbose=0), axis=1)
            y_pred_defended = np.argmax(defended_model.predict(x_test_adv_defended, verbose=0), axis=1)

        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        acc_defended = accuracy_score(y_test, y_pred_defended)
        baseline_accuracies.append(acc_baseline)
        defended_accuracies.append(acc_defended)
        print(f"Baseline Model Accuracy: {acc_baseline:.4f}")
        print(f"Defended Model Accuracy: {acc_defended:.4f}")

    results_df = pd.DataFrame({'Epsilon': epsilons, 'Baseline_Accuracy': baseline_accuracies, 'Defended_Accuracy': defended_accuracies})
    results_df.to_csv('../results/robustness_results.csv', index=False)
    print("\nRobustness results saved to ../results/robustness_results.csv")
    print("\nExperiment finished.")