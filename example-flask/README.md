# Running LLMs as Web App
This is same as the basic app except that it is being run with Flask as web app. 
The basic LLM query mechanism is the same as in the other example. For the GET method, very crude extraction and translation is done

## Running
* As a REST/json service
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{
  "system_message": "You are a helpful assistant",
  "user_message": "Generate a list of 5 funny dog names",
  "max_tokens": 100
  }' http://127.0.0.1:5000/llama 
  ```
* As web application
  Parameters: user_message, system_message, and max_tokens can be provided.
  Note: the web parameters are escaped by the browser.
  ```buildoutcfg
  http://localhost:5000/?user_message=list%20recipe%20for%20making%20chicken%20masala
  ```
  
## Docker
To build docker image, the model file will have to be physically copied to models/ dir
  ```buildoutcfg
  docker build . -t llama-flask
  docker run -p 5000:5000 llama-flask:latest
  ```

## References
* How to run Llama 2 locally on CPU + serving it as a Docker container | by Nikolay Penkov | Medium[[medium.com](https://medium.com/@penkow/how-to-run-llama-2-locally-on-cpu-docker-image-731eae6398d1)]
