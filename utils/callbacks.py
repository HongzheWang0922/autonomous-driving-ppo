
from stable_baselines3.common.callbacks import BaseCallback
import os

class DriveBackupCallback(BaseCallback):
    """
    训练过程中自动备份到Google Drive
    """
    def __init__(self, backup_freq=50000, backup_path='/content/drive/MyDrive/rl-project-backup', verbose=1):
        super().__init__(verbose)
        self.backup_freq = backup_freq
        self.backup_path = backup_path
        self.backup_count = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.backup_freq == 0:
            self.backup_count += 1
            if self.verbose > 0:
                print(f"\n{'='*50}")
                print(f"🔄 自动备份 #{self.backup_count}")
                print(f"{'='*50}")
            
            try:
                # 备份模型
                model_path = f"{self.backup_path}/models/checkpoint_{self.n_calls}"
                self.model.save(model_path)
                
                if self.verbose > 0:
                    print(f"✓ 模型已保存: checkpoint_{self.n_calls}.zip")
                    print(f"✓ 训练步数: {self.n_calls:,}")
                    print(f"{'='*50}\n")
                    
            except Exception as e:
                print(f"⚠️ 备份失败: {e}")
                
        return True

print("✓ utils/callbacks.py 创建成功")
