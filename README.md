## Python Environment Setup

#### pyenv
https://github.com/pyenv/pyenv

```sh
pyenv install 3.6.8
```

#### Virtualenv
https://sourabhbajaj.com/mac-setup/Python/virtualenv.html

```sh
pip install virtualenv
pip install virtualenvwrapper

mkvirtualenv -p /usr/local/python-3.6.7/bin/python sdc
```

#### Dev
```
# Activate virtualenv
workon sdc

# Install dependencies
pip install -r requirements.txt

# Start notebook
jupyter notebook
```
