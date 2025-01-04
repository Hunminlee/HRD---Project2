import matplotlib.pyplot as plt


def draw_fig_check_nan(nan_col_data, cnt_data):
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), constrained_layout=True)

    colors = ['red', 'blue', 'green', 'purple']

    for i in range(4):

        axes[i].bar(nan_col_data[i], cnt_data[i], color=colors[i])
        axes[i].set_title(f"Year 202{i} Dataset", fontsize=22)
        if i == 0:
            axes[i].set_ylabel("Count of NaN (Not a Number)", fontsize=18)
        axes[i].set_xticklabels(nan_col_data[i], rotation=45)
        axes[i].grid(True)

    plt.show()


def draw_fig_class_check(work_data_lst, target_lst):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    colors = ['red', 'blue', 'green', 'purple']

    for i in range(4):

        class_counts = work_data_lst[i][target_lst[i]].value_counts()

        axes[i].bar(class_counts.index, class_counts.values, color=colors[i])
        axes[i].set_title(f"Year 202{i + 1}", fontsize=25)
        axes[i].set_xlabel('Class', fontsize=20)
        if i == 0:
            axes[i].set_ylabel('Count', fontsize=20)
        axes[i].grid(True, linestyle='--')

    plt.show()


def draw_fig_class(new_lst):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)

    colors = ['red', 'blue', 'green', 'purple']

    for i in range(4):

        class_counts = new_lst[i].value_counts()

        axes[i].bar(class_counts.index, class_counts.values, color=colors[i])
        axes[i].set_title(f"Year 202{i + 1}", fontsize=25)
        axes[i].set_xlabel('Class', fontsize=20)
        if i == 0:
            axes[i].set_ylabel('Count', fontsize=20)
        axes[i].grid(True, linestyle='--')

    plt.show()


def overlap_check(year1_data, year2_data):
    cnt = 0
    for i in range(len(year1_data)):
        for j in range(len(year2_data)):
            if year1_data[i] == year2_data[j]:
                cnt += 1
                break
    print(
        f"Overlap data between two years : {cnt} samples out of Year1 : {len(year1_data)} and Year2: {len(year2_data)}")