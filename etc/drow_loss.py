import matplotlib.pyplot as plt

# ==========================================================
# 1. 터미널 창에 찍힌 Loss 값을 그대로 복사해서 여기에 넣으세요!
# ==========================================================

# [수정 전] ReLU 모델 학습 시 찍힌 Loss 리스트
ko_loss_relu = [6.431, 1.911, 1.896, 1.931, 2.095]  # 실제 값으로 변경
en_loss_relu = [10.062, 6.826, 6.975, 7.335, 7.686]  # 실제 값으로 변경

# [수정 후] GELU 모델 학습 시 찍힌 Loss 리스트
ko_loss_gelu = [10.074, 1.803, 1.635, 1.492, 1.199]  # 실제 값으로 변경
en_loss_gelu = [10.803, 4.087, 2.788, 1.700, 0.913]  # 실제 값으로 변경

# ==========================================================
# 2. 그래프 그리기 연산
# ==========================================================
epochs = range(1, len(ko_loss_relu) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 한국어 결과 차트
ax1.plot(epochs, ko_loss_relu, 'r--', marker='o', label='Baseline (ReLU)')
ax1.plot(epochs, ko_loss_gelu, 'b-', marker='s', label='Proposed (GELU)')
ax1.set_title('Korean Model Train Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 영어 결과 차트
ax2.plot(epochs, en_loss_relu, 'r--', marker='o', label='Baseline (ReLU)')
ax2.plot(epochs, en_loss_gelu, 'b-', marker='s', label='Proposed (GELU)')
ax2.set_title('English Model Train Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.suptitle('Activation Function Comparison: ReLU vs GELU', fontsize=14, fontweight='bold')
plt.tight_layout()

# 이미지 파일로 저장
plt.savefig('experiment_loss_comparison.png', dpi=200)
print("[성공] 'experiment_loss_comparison.png' 파일로 그래프가 저장되었습니다!")