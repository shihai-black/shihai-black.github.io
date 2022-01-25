# Hexo 分支备份法

> 换了家公司，新公司发的电脑，所以需要将写博客那一套迁移到新电脑

分支备份法：将hexo博客的原始内容放在分支中

好处：

1. 换电脑直接从分支从拉取文件，配置hexo即可操作
2. 一旦本地文件丢失，可以从何止中找回。

## 共同操作

```
# 配置全局操作
git config --global user.name "shihai-black"   
git config --global user.email "18358029413@163.com"
ssh-keygen -t rsa -C "18358029413@163.com"
ssh -T git@github.com 

# 安装hexo
npm install hexo
npm install
npm install hexo-deployer-git
```



## 具体操作

```csharp
# 共同操作走一波

mkdir hexo
cd hexo
git clone -b master git@github.com:shihai-black/shihai-black.github.io.git  #下载文件
rm -rf *
  
# 将备份文件放入
scaffolds/
source/
themes/
.git/
.gitignore
_config.yml
package.json

# git初始化
git init
#创建hexo分支，用来存放源码
git checkout -b hexo
#git 文件添加
git add --all
#git 提交
git commit -m "init"
#添加远程仓库
git remote set-url origin https://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@github.com/shihai-black/shihai-black.github.io.git
git remote git@github.com:shihai-black/shihai-black.github.io.git
#push到hexo分支
git push origin hexo

```

## 以后使用

```
# 安装hexo
npm install hexo
npm install
npm install hexo-deployer-git

# 从hexo下载原始文件
git clone -b hexo git@github.com:shihai-black/shihai-black.github.io.git  #下载文件 
```

## hexo常用操作

```
hexo help  # 查看帮助
hexo version  #查看Hexo的版本
hexo algolia  # 更新search庫
hexo new "postName" #新建文章
hexo new post "title"  # 生成新文章：\source\_posts\title.md，可省略post
hexo new page "pageName" #新建页面
hexo clean #清除部署緩存
hexo n == hexo new #新建文章
hexo g == hexo generate #生成静态页面至public目录
hexo s == hexo server #开启预览访问端口（默认端口4000，'ctrl + c'关闭server）
hexo d == hexo deploy #将.deploy目录部署到GitHub
hexo d -g #生成加部署
hexo s -g #生成加预览
```

## reference

https://www.jianshu.com/p/aebeaf050969

https://www.jianshu.com/p/c058fbd7bb90

## hexo显示需要密码
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