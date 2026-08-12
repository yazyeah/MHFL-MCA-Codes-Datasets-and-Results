import os

# 设置环境变量
os.environ["TEMP"] = "D:\\temp"
os.environ["TMP"] = "D:\\temp"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten, Conv1D, MaxPooling1D, AveragePooling1D, Lambda, Concatenate, \
    Activation, LeakyReLU
import keras.backend as K
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc, accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from itertools import cycle
import optuna

# [全局绘图设置] 300 DPI, Times New Roman 字体
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 兼容 Optuna Integration
try:
    from optuna_integration import TFKerasPruningCallback
except ImportError:
    try:
        from optuna.integration import TFKerasPruningCallback
    except ImportError:
        TFKerasPruningCallback = None

# ================= 1. 全局配置 =================

# 实验配置
SEARCH_TRIALS = 30
SAMPLE_RANGE = range(5, 31)
REPEAT_TIMES = 10

DATA_POINTS = 2048
NUM_CLASSES = 7

# [修改点] 路径保持不变
BASE_OUTPUT_DIR = r"D:\机械故障诊断实验\MHCNN_Optuna_Experiment"
DATA_PATH_ROOT = r"D:\渥太华数据\3_MatLab_Raw_Data"


# ================= 2. 数据加载 =================

def load_all_data():
    print("正在加载数据...")
    tot_num0 = 200

    def load_mat_category(folder, filename):
        path = os.path.join(DATA_PATH_ROOT, folder, filename)
        try:
            data = loadmat(path)
            key = filename.replace('.mat', '')
            if key not in data:
                keys = [k for k in data.keys() if not k.startswith('__')]
                if keys: key = keys[0]
            val = data[key]
            vib = val[:, 0];
            aco = val[:, 1]
            vib_samples = np.zeros((tot_num0, DATA_POINTS))
            aco_samples = np.zeros((tot_num0, DATA_POINTS))
            for i in range(tot_num0):
                if (i + 1) * DATA_POINTS <= len(vib):
                    vib_samples[i, :] = vib[i * DATA_POINTS:(i + 1) * DATA_POINTS]
                    aco_samples[i, :] = aco[i * DATA_POINTS:(i + 1) * DATA_POINTS]
            return vib_samples, aco_samples
        except Exception as e:
            print(f"加载错误 {path}: {e}")
            return np.zeros((tot_num0, DATA_POINTS)), np.zeros((tot_num0, DATA_POINTS))

    datasets = []

    def load_class_data(file_list):
        v_list, a_list = [], []
        for folder, file in file_list:
            v, a = load_mat_category(folder, file)
            v_list.append(v);
            a_list.append(a)
        return np.vstack(v_list), np.vstack(a_list)

    datasets.append(load_class_data([("1_Healthy", "H_1_0.mat"), ("1_Healthy", "H_2_0.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_1.mat"), ("2_Inner_Race_Faults", "I_2_1.mat")]))
    datasets.append(load_class_data([("2_Inner_Race_Faults", "I_1_2.mat"), ("2_Inner_Race_Faults", "I_2_2.mat")]))
    datasets.append(load_class_data([("3_Outer_Race_Faults", "O_6_2.mat"), ("3_Outer_Race_Faults", "O_7_2.mat")]))
    datasets.append(load_class_data([("4_Ball_Faults", "B_11_2.mat"), ("4_Ball_Faults", "B_12_2.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_1.mat"), ("5_Cage_Faults", "C_17_1.mat")]))
    datasets.append(load_class_data([("5_Cage_Faults", "C_16_2.mat"), ("5_Cage_Faults", "C_17_2.mat")]))

    print("数据加载完成。")
    return datasets


ALL_DATASETS = load_all_data()


def get_split_data(datasets, seed, num_train_per_class):
    train_list, test_list = [], []
    for label, (vib, aco) in enumerate(datasets):
        combined = np.hstack((vib, aco))
        np.random.seed(seed)
        np.random.shuffle(combined)
        train_part = combined[:num_train_per_class, :]
        test_part = combined[num_train_per_class:, :]
        y_train = np.full((len(train_part), 1), label)
        y_test = np.full((len(test_part), 1), label)
        train_list.append(np.hstack((train_part, y_train)))
        test_list.append(np.hstack((test_part, y_test)))
    train_all = np.vstack(train_list);
    test_all = np.vstack(test_list)
    np.random.shuffle(train_all);
    np.random.shuffle(test_all)
    x_train_vib = train_all[:, 0:2048][:, :, np.newaxis]
    x_train_aco = train_all[:, 2048:4096][:, :, np.newaxis]
    y_train = np_utils.to_categorical(train_all[:, 4096], NUM_CLASSES)
    x_test_vib = test_all[:, 0:2048][:, :, np.newaxis]
    x_test_aco = test_all[:, 2048:4096][:, :, np.newaxis]
    y_test = np_utils.to_categorical(test_all[:, 4096], NUM_CLASSES)
    return (x_train_vib, x_train_aco, y_train), (x_test_vib, x_test_aco, y_test)


# ================= 3. 模型定义 =================

class CrossAttention(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CrossAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name='W_q', shape=(input_shape[0][-1], self.output_dim), initializer='uniform',
                                   trainable=True)
        self.W_k = self.add_weight(name='W_k', shape=(input_shape[1][-1], self.output_dim), initializer='uniform',
                                   trainable=True)
        self.W_v = self.add_weight(name='W_v', shape=(input_shape[1][-1], self.output_dim), initializer='uniform',
                                   trainable=True)
        super(CrossAttention, self).build(input_shape)

    def call(self, inputs):
        query, key = inputs
        query_proj = K.dot(query, self.W_q)
        key_proj = K.dot(key, self.W_k)
        value_proj = K.dot(key, self.W_v)
        attention_scores = K.batch_dot(query_proj, key_proj, axes=[2, 2])
        attention_scores = K.softmax(attention_scores, axis=-1)
        return K.batch_dot(attention_scores, value_proj)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def weighted_features(inputs):
    features, weights = inputs
    return features * weights


def create_model(trial=None):
    if trial:
        dropout_vib = trial.suggest_float("dropout_vib", 0.1, 0.5)
        dropout_aco = trial.suggest_float("dropout_aco", 0.1, 0.5)
        atten_dim = trial.suggest_categorical("atten_dim", [128, 256])
        n_layers_vib = trial.suggest_int("n_layers_vib", 3, 5)
        n_layers_aco = trial.suggest_int("n_layers_aco", 3, 5)
    else:
        dropout_vib = 0.2;
        dropout_aco = 0.4;
        atten_dim = 256;
        n_layers_vib = 4;
        n_layers_aco = 4

    alpha = 0.2

    # --- 振动分支 ---
    input_a = Input(shape=(2048, 1))
    x = input_a
    filters_vib = [32, 64, 128, 256, 256]
    for i in range(n_layers_vib):
        f = filters_vib[i] if i < len(filters_vib) else 256
        stride = 2 if i < 2 else 1
        kernel = 16
        x = Conv1D(f, kernel, strides=stride, padding='same')(x)
        x = LeakyReLU(alpha=alpha)(x)
        if i == n_layers_vib - 1:
            x = Dropout(dropout_vib)(x)
            features1 = AveragePooling1D(2, strides=2, padding='same')(x)
        else:
            x = AveragePooling1D(2, strides=2, padding='same')(x)

    # --- 声学分支 ---
    input_b = Input(shape=(2048, 1))
    y = input_b
    filters_aco = [32, 64, 128, 256, 256]
    for i in range(n_layers_aco):
        f = filters_aco[i] if i < len(filters_aco) else 256
        stride = 2 if i < 2 else 1
        kernel = 8
        y = Conv1D(f, kernel, strides=stride, padding='same')(y)
        y = Activation('gelu')(y)
        if i == n_layers_aco - 1:
            y = Dropout(dropout_aco)(y)
            features2 = MaxPooling1D(2, strides=2, padding='same')(y)
        else:
            y = MaxPooling1D(2, strides=2, padding='same')(y)

    cross_att_1 = CrossAttention(output_dim=atten_dim)
    features_b_att = cross_att_1([features1, features2])
    cross_att_2 = CrossAttention(output_dim=atten_dim)
    features_a_att = cross_att_2([features2, features1])

    flat_a = Flatten()(features_a_att)
    flat_b = Flatten()(features_b_att)

    w_final = Dense(2, activation='softmax')(
        Concatenate()([Dense(1, activation='sigmoid')(flat_a), Dense(1, activation='sigmoid')(flat_b)]))
    fused = Concatenate()(
        [Lambda(weighted_features)([flat_a, w_final[:, 0:1]]), Lambda(weighted_features)([flat_b, w_final[:, 1:2]])])
    fused = LeakyReLU(alpha=0.2)(Dense(256)(fused))
    output = Dense(NUM_CLASSES, activation='softmax')(fused)

    return Model(inputs=[input_a, input_b], outputs=output), Model(inputs=[input_a, input_b], outputs=fused)


# ================= 4. Optuna 目标函数 =================

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(ALL_DATASETS, seed=42, num_train_per_class=15)

    try:
        model, _ = create_model(trial)
        model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=lr), loss='categorical_crossentropy',
                      metrics=['accuracy'])
        callbacks = [TFKerasPruningCallback(trial, "val_accuracy")] if TFKerasPruningCallback else []
        history = model.fit([x_tr_v, x_tr_a], y_tr, epochs=25, batch_size=batch_size,
                            validation_data=([x_te_v, x_te_a], y_te), verbose=0, callbacks=callbacks)
        return history.history['val_accuracy'][-1]
    except ValueError:
        return 0.0


# ================= 5. 主程序 =================

if __name__ == "__main__":

    # --- 阶段 1: Optuna ---
    print(">>> 阶段 1: Optuna 搜索...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=SEARCH_TRIALS)
    print("最佳参数:", study.best_params)

    from optuna.trial import FixedTrial

    best_trial_context = FixedTrial(study.best_params)
    best_lr = study.best_params['lr']
    best_bs = study.best_params['batch_size']

    # --- [新增] 打印最优模型结构 ---
    print("\n>>> Best MHCNN Model Architecture Summary:")
    print("=" * 60)
    try:
        # 创建一个临时模型实例来打印摘要
        temp_model, _ = create_model(best_trial_context)
        temp_model.summary()
        # 释放内存
        del temp_model
    except Exception as e:
        print(f"Error printing model summary: {e}")
    print("=" * 60 + "\n")

    # --- 阶段 2: 全面实验 ---
    print("\n>>> 阶段 2: 全样本范围敏感性测试 (5-30 samples)...")
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    summary_stats = []

    for num_samples in SAMPLE_RANGE:
        sample_dir = os.path.join(BASE_OUTPUT_DIR, f"Samples_{num_samples:02d}")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"\n======== 测试样本量: {num_samples} (Run 1-{REPEAT_TIMES}) ========")

        metrics_buffer = {'acc': [], 'f1': [], 'prec': [], 'recall': []}

        for run_idx in range(1, REPEAT_TIMES + 1):
            run_dir = os.path.join(sample_dir, f"Run_{run_idx:02d}")
            os.makedirs(run_dir, exist_ok=True)

            # 数据
            seed = num_samples * 100 + run_idx
            (x_tr_v, x_tr_a, y_tr), (x_te_v, x_te_a, y_te) = get_split_data(ALL_DATASETS, seed=seed,
                                                                            num_train_per_class=num_samples)

            # 训练
            model, feat_model = create_model(best_trial_context)
            model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=best_lr), loss='categorical_crossentropy',
                          metrics=['accuracy'])
            history = model.fit([x_tr_v, x_tr_a], y_tr, epochs=80, batch_size=best_bs,
                                validation_data=([x_te_v, x_te_a], y_te), verbose=0)

            # 预测
            y_pred_prob = model.predict([x_te_v, x_te_a], verbose=0)
            y_pred = np.argmax(y_pred_prob, axis=1)
            y_true = np.argmax(y_te, axis=1)

            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            prec = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')

            metrics_buffer['acc'].append(acc);
            metrics_buffer['f1'].append(f1)
            metrics_buffer['prec'].append(prec);
            metrics_buffer['recall'].append(rec)

            print(f"  [S{num_samples}-R{run_idx}] Acc: {acc:.4f}, F1: {f1:.4f}")

            # --- 可视化 ---
            # (A) Curves
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1);
            plt.plot(history.history['loss'], label='Train');
            plt.plot(history.history['val_loss'], label='Test')
            plt.title(f'Loss - Sample {num_samples}');
            plt.legend()
            plt.subplot(1, 2, 2);
            plt.plot(history.history['accuracy'], label='Train');
            plt.plot(history.history['val_accuracy'], label='Test')
            plt.title(f'Accuracy - Sample {num_samples}');
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'curves.png'), dpi=300)
            plt.close()

            # (B) CM
            plt.figure(figsize=(6, 5))
            sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Sample {num_samples}')
            plt.savefig(os.path.join(run_dir, 'cm.png'), dpi=300)
            plt.close()

            # (C) t-SNE
            feats = feat_model.predict([x_te_v, x_te_a], verbose=0)
            tsne = TSNE(n_components=2, random_state=42).fit_transform(feats)
            plt.figure(figsize=(6, 5))
            sns.scatterplot(x=tsne[:, 0], y=tsne[:, 1], hue=y_true, palette='tab10', legend='full', s=15)
            plt.title(f't-SNE - Sample {num_samples}')
            plt.savefig(os.path.join(run_dir, 'tsne.png'), dpi=300)
            plt.close()

            # (D) ROC
            y_test_bin = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
            fpr, tpr, roc_auc = dict(), dict(), dict()
            for i in range(NUM_CLASSES):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(NUM_CLASSES):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= NUM_CLASSES
            fpr["macro"] = all_fpr;
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            plt.figure(figsize=(8, 6))
            plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg (AUC={roc_auc["micro"]:.4f})', color='deeppink',
                     linestyle=':', lw=3)
            plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-avg (AUC={roc_auc["macro"]:.4f})', color='navy',
                     linestyle=':', lw=3)
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
            for i, color in zip(range(NUM_CLASSES), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC={roc_auc[i]:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate');
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Sample {num_samples}')
            plt.legend(loc="lower right", fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, 'roc.png'), dpi=300)
            plt.close()

        # [NEW LOGIC] 去掉最高最低，取平均
        stats = [num_samples]
        for metric in ['acc', 'f1', 'prec', 'recall']:
            values = sorted(metrics_buffer[metric])
            # 去掉最低和最高 (切片 [1:-1])
            if len(values) > 2:
                trimmed_values = values[1:-1]
            else:
                trimmed_values = values  # 如果次数太少就不切了

            stats.append(np.mean(trimmed_values))
            stats.append(np.std(trimmed_values))

        summary_stats.append(stats)
        print(f"  >>> S{num_samples} 完成 (Trimmed): Mean Acc={stats[1]:.4f} (Std={stats[2]:.4f})")

    cols = ['Samples', 'Acc_Mean', 'Acc_Std', 'F1_Mean', 'F1_Std', 'Prec_Mean', 'Prec_Std', 'Recall_Mean', 'Recall_Std']
    df = pd.DataFrame(summary_stats, columns=cols)
    df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'Final_Summary_Stats.csv'), index=False)

    # 趋势图
    plt.figure(figsize=(10, 6))
    plt.errorbar(df['Samples'], df['Acc_Mean'], yerr=df['Acc_Std'], fmt='-o', label='Accuracy', capsize=5)
    plt.errorbar(df['Samples'], df['F1_Mean'], yerr=df['F1_Std'], fmt='-s', label='F1 Score', capsize=5)
    plt.xlabel('Training Samples per Class');
    plt.ylabel('Score')
    plt.title('Performance vs. Sample Size')
    plt.grid(True, alpha=0.5);
    plt.legend()
    plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'performance_trend.png'), dpi=300)
    print(f"全部完成！结果已保存在: {BASE_OUTPUT_DIR}")