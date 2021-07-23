import configparser, json, io, os

class Config():
    """
    A config object that can be created in four diffrent manners. Paramaters 
    can be input as a string, stream, file or dict. By default a dict is 
    assumed. To input string, stream or file use the respective class 
    methods; from_string, from_stream, from_file.
    """

    @classmethod
    def from_string(cls,string,from_default=False) -> object:
        """
        Create config object from string
        Paramaters:
            string: str object in either json or ini format 
            from_default: Use default values for values not specified
                          (requires an implementation of the abstract 
                          set_default method). Default False.
        Returns:
            Config object
        """
        #Make string into IO stream and parse as stream instead
        return cls.from_stream(io.StringIO(string), from_default)

    @classmethod
    def from_file(cls,path,from_default=False) -> object:
        """
        Create config object from file
        Paramaters:
            path: str object, path to either json or ini file 
            from_default: Use default values for values not specified
                          (requires an implementation of the abstract 
                          set_default method). Default False.
        Returns:
            Config object
        """
        #Open file as stream and parse as stream instead
        with open(path,'r') as buf:
            return cls.from_stream(buf, from_default)

    @classmethod
    def from_stream(cls,stream,from_default=False) -> object:
        """
        Create config object from IO stream
        Paramaters:
            stream: file like object in either json or ini format
            from_default: Use default values for values not specified
                          (requires an implementation of the abstract 
                          set_default method). Default False.
        Returns:
            Config object
        """
        #If default values read them
        params = cls.get_default() if from_default else {}
        #Try to load json file
        try:
            #Create object from json stream
            params.update(json.load(stream))
            #Values as strings is false for json streams
            vas = False
        #If not json file parse as INI file
        except json.decoder.JSONDecodeError:
            stream.seek(0)
            #Initialize configparser object
            config = configparser.ConfigParser()
            #Read config from stream
            config.read_file(stream)
            #Get header title
            header = config.sections()[0]
            #Update default values based on INI string
            params.update(config._sections[header])
            #Values as strings is true for ini streams
            vas = True
            #Create config from INI stream
        return cls(params, values_as_strings = vas)
    
    @classmethod
    def from_dict(cls, dictionary, from_default = False) -> object:
        """
        Create config from dictionary
        Paramaters:
            dictionary: dict type object with names of paramaters as keys and 
                        values as values
            from_default: Use default values for values not specified
                          (requires an implementation of the abstract 
                          set_default method). Default False.
        """
        params = cls.get_default() if from_default else {}
        params.update(dictionary)
        return cls(params)
    
    @classmethod
    def from_default(cls) -> object:
        """
        Create config object from default values. This requires 
        that get_default is implemented.

        Returns:
            Config object
        """
        #Get default values
        params = cls.get_default()
        #Create config object from default values
        return cls(params)

    @classmethod
    def get_default(cls) -> dict:
        """
        Set default value of config class. Abstract method intended to be
        overidden by subclasses.

        Returns:
            Dictionary with default values
        """
        raise NotImplementedError('Abstract method ment to be overidden by subclass')
        

    def __init__(self, paramaters, values_as_strings = False) -> None:
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
        self.values_as_strings = values_as_strings

    def get_keys(self) -> list:
        """
        Getter for paramater keys used for accsessing paramaters

        Returns:
            List of keys in paramater dictionary
        """
        return list(self._params.keys())
    
    def set_param(self, param, value) -> None:
        """
        Setter for paramters, allow editing paramaters after initilization
        Paramaters:
            param: str object with the name of the paramater to change
            value: new value of paramater
        """

        self._params[param] = value
    
    def save_param(self, filename, output_dir=None, ini=False) -> None:
        """
        Save paramaters to file.
        Paramaters:
            filename: Name of file to store paramaters in
            outputdir: Optional, specify directory to store paramaterfile in.
                       The function will attempt to create this directory if 
                       it is not present.
            ini: Defaults to false, specify wether the user wants to write to 
                  ini file, the default type is json.
        """
        if output_dir is None:
            path = filename
        else:
            #If output dir specified create the directory
            os.system("mkdir -p %s"%(output_dir))
            path = output_dir + '/' + filename

        if ini:
            path += '.ini'
            #Parse directory into inifile using configparser
            c = configparser.ConfigParser()
            headerdict = {'config': self._params}
            c.read_dict(headerdict)
            with open(path, 'w') as file:
                c.write(file)
        else:
            path += '.json'
            #Parse directory into json file using configparser
            with open(path,'w') as file:
                json.dump(self._params,file, sort_keys=True, indent=4)


    def __getitem__(self, param) -> any:
        """
        Overide getitem such that class behaves as a dictionary
        Paramaters:
            param: str object with the name of the paramater to get
        """
        return self._params[param]
    
    def __str__(self) -> str:
        return "Config object:\n" + json.dumps(self._params, indent=2)

    def __repr__(self) -> str:
        return "Config" + json.dumps(self._params)

