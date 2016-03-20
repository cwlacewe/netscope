Parser = require './parser'
Network = require '../network.coffee'

generateLayers = (descriptors, phase) ->
    phase ?= 'train'
    layers = []
    for entry in descriptors
        # Support the deprecated Caffe 'layers' key as well.
        layerDesc = entry.layer or entry.layers
        if layerDesc?
            layer = {}
            headerKeys = ['name', 'type', 'top', 'bottom']
            _.extend layer, _.pick(layerDesc, headerKeys)
            layer.attribs = _.omit layerDesc, headerKeys
            layers.push layer
        else
            console.log 'Unidentified entry ignored: ', entry
    layers = _.filter layers, (layer) ->
        layerPhase = layer.attribs.include?.phase
        not (layerPhase? and layerPhase!=phase)
    return layers

analyzeNetwork = (net) ->
    ## Add Input/Output Dimensions + Channels to each Node / Layer
    #shape.dim: (    N   x   K   x   W   x   H   )
    #             batch   channel  width   height
    
    for n in net.nodes
        # init to zero
        d = n.dim
        d.wIn = d.hIn = d.wOut = d.hOut = 0
        d.featIn = d.featOut = 0
        
        prev = n.parents[0]?.dim
        layertype = n.type.toUpperCase()
        switch layertype
            when "DATA"
                d.featIn = n.attribs.input_param.shape.dim[1]
                d.featOut = d.featIn
                d.wIn = n.attribs.input_param.shape.dim[2]
                d.hIn = n.attribs.input_param.shape.dim[3]
                d.wOut = d.wIn; d.hOut = d.hIn
                
            when "CONVOLUTION"
                kernel = n.attribs.convolution_param.kernel_size
                stride = n.attribs.convolution_param.stride ? 1
                pad    = n.attribs.convolution_param.pad ? 0
                numout = n.attribs.convolution_param.num_output
                d.wIn = prev.wOut; d.hIn = prev.hOut
                # according to http://caffe.berkeleyvision.org/tutorial/layers.html
                d.wOut = ((d.wIn + 2*pad - kernel) / stride + 1)
                d.hOut = ((d.hIn + 2*pad - kernel) / stride + 1)
                d.featIn = prev.featOut
                d.featOut = numout
                
            when "POOLING"
                kernel = n.attribs.pooling_param.kernel_size
                stride = n.attribs.pooling_param.stride ? 1
                pad    = n.attribs.pooling_param.pad ? 0
                isglobal = n.attribs.pooling_param.global_pooling ? 0
                d.wIn = prev.wOut; d.hIn = prev.hOut
                # according to http://caffe.berkeleyvision.org/tutorial/layers.html
                if !isglobal
                    d.wOut = ((d.wIn + 2*pad - kernel) / stride + 1)
                    d.hOut = ((d.hIn + 2*pad - kernel) / stride + 1)
                else
                    d.wOut = d.hOut = 1
                
                d.featIn = prev.featOut
                d.featOut = d.featIn
            
            when "CONCAT"
                d.wIn = prev.wOut; d.hIn = prev.hOut
                d.wOut = d.wIn; d.hOut = d.hIn
                
                # check all input dims agree
                dims_ok = true
                dims_ok = dims_ok && (p.dim.wOut == d.wIn & p.dim.hOut == d.hIn) for p in n.parents
                console.warn('CONCAT: input dimensions dont agree!') if not dims_ok
                
                # sum up channels from inputs
                d.featIn += p.dim.featOut for p in n.parents
                d.featOut = d.featIn
                
            else # RELU or unknown layer;  Out Dim = In Dim
                d.wIn = prev?.wOut;
                d.hIn = prev?.hOut
                d.wOut = d.wIn; d.hOut = d.hIn
                d.featIn = prev?.featOut
                d.featOut = d.featIn
                
        # add dimensions to node attributes
        # so they show in graph tooltips
        if (layertype!="RELU" && layertype!="SOFTMAX" && layertype!="SOFTMAXWITHLOSS")
            _.extend(n.attribs, {
            analysis: {
                in: d.featIn+'ch ⋅ '+d.wIn+'×'+d.hIn,
                out: d.featOut+'ch ⋅ '+d.wOut+'×'+d.hOut
                }} )
                
    return net

generateNetwork = (layers, header) ->
    nodeTable = {}
    implicitLayers = []
    net = new Network header.name
    getSingleNode = (name) =>
        node = nodeTable[name]
        # Caffe allows top to be a layer which isn't explicitly
        # defined. Create an implicit layer if this is detected.
        if not node?
            node = net.createNode name, 'implicit'
            nodeTable[name] = node
        return node
    getNodes = (names, exclude) =>
        names = [].concat names
        if exclude?
            _.pullAll names, exclude
        _.map names, getSingleNode
    # Build the node LUT.
    for layer in layers
        nodeTable[layer.name] = net.createNode layer.name, layer.type, layer.attribs, {}
    # Connect layers.
    inplaceTable = {}
    for layer in layers
        node = nodeTable[layer.name]
        if layer.top?
            if layer.top==layer.bottom
                # This is an inplace node. We will treat this specially.
                # Note that this would have otherwise introduced a cycle,
                # violating the requirements of a DAG.
                if not inplaceTable[layer.top]?
                    inplaceTable[layer.top] = []
                inplaceTable[layer.top].push node
                continue
            else
                node.addChildren getNodes(layer.top, [layer.name])
        if layer.bottom?
            node.addParents getNodes(layer.bottom, [].concat layer.top)
    # Splice in the inplace nodes.
    for own k, inplaceOps of inplaceTable
        curNode = nodeTable[k]
        curNode.coalesce = inplaceOps
        children = curNode.detachChildren()
        for inplaceChild in inplaceOps
            inplaceChild.annotation = 'InPlace'
            curNode.addChild inplaceChild
            curNode = inplaceChild
        curNode.addChildren children
    # Patch in data layer parameters.
    if header?.input? and header?.input_dim?
        inputs = [].concat header.input
        dims = header.input_dim
        if inputs.length==(dims.length*0.25)
            for input, i in inputs
                dataNode = nodeTable[input]
                dataNode.type = 'data'
                dataNode.attribs.shape = dims.slice i*4, (i+1)*4
        else
            console.log 'Inconsistent input dimensions.'
    return net

module.exports =
class CaffeParser
    @parse : (txt, phase) ->
        [header, layerDesc] = Parser.parse txt
        layers = generateLayers layerDesc, phase
        network = generateNetwork layers, header
        network = analyzeNetwork network
        return network
