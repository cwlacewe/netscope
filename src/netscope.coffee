AppController   = require './app.coffee'
CaffeNetwork    = require './caffe/caffe.coffee'
Loader          = require './loader.coffee'

showDocumentation = ->
    window.location.href = 'quickstart.html'

$(document).ready ->
    app = new AppController()
    # Setup Caffe model loader.
    # This can be replaced with any arbitrary parser to support
    # formats other than Caffe.
    loader = new Loader(CaffeNetwork)
    # Helper function for wrapping the load calls.
    makeLoader = (loadingFunc, loader) ->
        (args...) ->
            app.startLoading loadingFunc, loader, args...

    # Register routes
    routes =
       '/gist/:gistID' : makeLoader loader.fromGist, loader
       '/url/(.+)'     : makeLoader loader.fromURL, loader
       '/preset/:name' : makeLoader loader.fromPreset, loader
       '/editor(/?)'   : => app.showEditor loader
       '/doc'          : => showDocumentation()
    router = Router(routes)
    router.init '/doc'