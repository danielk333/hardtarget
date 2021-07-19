import configparser, json, io
from json.decoder import JSONDecodeError

class Config():
    """
    A config object that can be created in four diffrent manners. Paramaters 
    can be input as a string, stream, file or dict. By default a dict is 
    assumed. To input string, stream or file use the respective class 
    methods; from_string, from_stream, from_file.
    """

    @classmethod
    def from_string(cls,string) -> object:
        """
        Create config object from string
        Paramaters:
            string: str object in either json or ini format 
        Returns:
            Config object
        """
        try:
            return cls(json.loads(string))
        except json.decoder.JSONDecodeError:
            #Initialize configparser object
            config = configparser.ConfigParser()
            #Read config from string
            config.read_string(string)
            #Create config from INI string
            return cls(config._sections)

    @classmethod
    def from_file(cls, path) -> object:
        """
        Create config object from file
        Paramaters:
            path: str object, path to either json or ini file 
        Returns:
            Config object
        """
        #Open file from path
        try:
            with open(path,'r') as buf:
                #Create object from json file
                return cls(json.load(buf))
        except json.decoder.JSONDecodeError:
            #Initialize configparser object
            config = configparser.ConfigParser()
            #Read config from file
            config.read(path)
            #Create config from INI file
            return cls(config._sections)

    @classmethod
    def from_stream(cls,stream) -> object:
        """
        Create config object from IO stream
        Paramaters:
            stream: file like object in either json or ini format 
        Returns:
            Config object
        """
        #Try to load json file
        try:
            #Create object from json stream
            return cls(json.load(stream))
        #If not json file parse as INI file
        except json.decoder.JSONDecodeError:
            stream.seek(0)
            #Initialize configparser object
            config = configparser.ConfigParser()
            #Read config from stream
            config.read_file(stream)
            #Create config from INI stream
            return cls(config._sections)
        

    def __init__(self, paramaters) -> object:
        """
        A config object that can be created in four diffrent manners. Paramaters 
        can be input as a string, stream, file or dict. By default a dict is 
        assumed. To input string, stream or file use the respective class 
        methods; from_string, from_stream, from_file.

        Paramaters:
            paramaters: dict type object with names of paramaters as keys and 
                        values as values
        """
        self._params = paramaters

    def get_keys(self):
        """
        Getter for paramater keys used for accsessing paramaters

        Returns:
            List of keys in paramater dictionary
        """
        return self._params.keys()
    
    def set_param(self, param, value) -> None:
        """
        Setter for paramters, allow editing paramaters after initilization
        Paramaters:
            param: str object with the name of the paramater to change
            value: new value of paramater
        """

        self._params[param] = value

    def __getitem__(self, param) -> any:
        """
        Overide getitem such that class behaves as a dictionary
        Paramaters:
            param: str object with the name of the paramater to get
        """
        return self._params[param]

