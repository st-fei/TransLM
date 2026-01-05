# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def main():
    '''
    llamafactory-cli核心入口
    '''
    from .extras.misc import is_env_enabled
    # 若USE_V1环境变量为True，则使用v1版本的launcher（对应v1文件夹）
    if is_env_enabled("USE_V1"):
        from .v1 import launcher
    # 否则，使用当前目录下的launcher
    else:
        from . import launcher
    # 调用launch，执行实际的命令行处理逻辑
    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    # freeze_support用于保证跨平台兼容性
    freeze_support()
    main()
