# dynamic-risk-assessment-system
Dynamic Risk Assessment System


# Creating environment

- for this project, I made use of Ubuntu 22 on AWS t3.medium EC2 instances with  30 GB of Storage    

to connect the VSCODE, install the 'remote explorer' extension and configure the .ssh config file  
>Host MLOPS2-EC2-UBUNTU   
    HostName ec2-35-170-79-232.compute-1.amazonaws.com  
    User ubuntu  
    IdentityFile /Users/mnave/.ssh/MLOPS2.pem  

- create environment as follows and then activate  
`python3 -m vevn .venv  # this creates the environment`

- Activate the environment  
`source ./.venv/bin/activate`  

Alternately, add "`source ./.venv/bin/activate`" as a last line in **.bashrc file**. This activates the environment everytime the terminal is opened  

#### Troubleshooting
in case of issue in creating environment. following the following steps  
`sudo apt-get update`  
`sudo apt install python3-dev`  
`sudo apt install python3-venv`


### Clone the repository
`git clone https://github.com/mdeevan/dynamic-risk-assessment-system.git`



DAGSHUB will be used to track the data and model versioning (Data version control), while GITHUB will be used for the code   
MLProject -> project file for the MLFlow  
python_env.yaml -> contains the python environment configuration  
requirement.txt -> containing the requirements data  

