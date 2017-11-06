Renderer    = require './renderer.coffee'
Editor      = require './editor.coffee'

module.exports =
class AppController
    constructor: ->
        @inProgress = false
        @$spinner   = $('#net-spinner')
        @$netBox    = $('#net-container')
        @$netError  = $('#net-error')
        @svg        = '#net-svg'
        @$tableBox  = $('#table-container')
        @table      = '#table-content'
        @setupErrorHandler()

    startLoading: (loaderFunc, loader, args...) ->
        if @inProgress
            return
        @$netError.hide()
        @$netBox.hide()
        @$tableBox.hide()
        @$spinner.show()
        loaderFunc args..., (net) => @completeLoading(net, loader)

    completeLoading: (net, loader) ->
        @$spinner.hide()

        $('#net-title').html(net.name.replace(/_/g, ' '))

        $('title').text(net.name.replace(/_/g, ' ')+' — Netscope CNN Analyzer')
        # editlink = $("<a>(edit)</a>").addClass("editlink")
        # editlink.appendTo $('#net-title')
        # editlink.click( => @showEditor(loader))
        @showEditor(loader)

        @$netBox.show()
        @$tableBox.show()
        $(@svg).empty()
        $('.qtip').remove()
        @renderer = new Renderer net, @svg, @table

        if not window.do_variants_analysis
            $("<br>").appendTo @table 
            extendlink = $('<a>Excel-compatible Analysis Results (experimental)</a>')
            extendlink.click( => 
                window.do_variants_analysis = true
                @renderer.renderTable()
            )
            extendlink.appendTo @table 

        @inProgress = false

    makeLoader: (loaderFunc, loader) ->
        (args...) =>
            @startLoading loaderFunc, loader, args...

    showEditor: (loader) ->
        # Display the editor by lazily loading CodeMirror.
        # loader is an instance of a Loader.
        if(_.isUndefined(window.CodeMirror))
            $.getScript 'assets/js/lib/codemirror.min.js', =>
                @netEditor = new Editor(@makeLoader(loader.load, loader), loader)
        else
            @netEditor.reload(loader.load, loader)

    setupErrorHandler: ->
        window.onerror = (message, filename, lineno, colno, e) =>
            msg = message
            if not (_.isUndefined(e) || _.isUndefined(e.line) || _.isUndefined(e.column))
                msg = _.template('Line ${line}, Column ${column}: ${message}')(e)
            @$spinner.hide()
            $('.msg', @$netError).html(msg);
            @$netError.show()
            @inProgress = false

