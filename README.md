# Netscope CNN Analyzer

Available here: http://cwlacewe.github.io/netscope 

This is a CNN Analyzer tool, based on the Netscope fork by [dgschwend](https://github.com/dgschwend) which was originally by [ethereon](https://github.com/ethereon).

Netscope is a web-based tool for visualizing neural network topologies. It currently supports UC Berkeley's [Caffe framework](https://github.com/bvlc/caffe).

This version of netscope was tested in Visual Studio Code on Windows 10 with Node.js(v9.3.0)

## Documentation

Netscope [Quick Start Guide](http://cwlacewe.github.io/netscope/quickstart.html)

## Installation

After cloning this repo, you can obtain your updates via http-server using the following directions:

1. Install Node.js from the [Node.js website](https://nodejs.org/en/).
1. Run the installer and make sure the npm package manager is included.
1. Install http-server using:
    ```bash
    npm install http-server -g
    ```
1. Start server
    ```bash
    cd netscope
    http-server
    ```
1. Open in the browser at http://localhost:8080

## Demo

[Visualization of ResNet-50](http://cwlacewe.github.io/netscope/#/preset/resnet-50)

## License

Released under the MIT license.
All included network models provided under their respective licenses.
