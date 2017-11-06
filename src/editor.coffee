module.exports =
class Editor
    constructor: (@loaderFunc, loader) ->
        editorWidthPercentage = 30;
        $editorBox = $($.parseHTML '<div id="editor-col" class="column"></div>')
        $editorBox.width(editorWidthPercentage+'%')
        $('#net-column').width((100-editorWidthPercentage)+'%')
        $('#master-container').prepend $editorBox
        preset = loader.dataLoaded ? '# Enter your network definition here.\n# Use Shift+Enter to update the visualization.'
        @editor = CodeMirror $editorBox[0],
            value: preset
            lineNumbers : true
            lineWrapping : true
        @editor.on 'keydown', (cm, e) => @onKeyDown(e)

    reload: (@loaderFunc, loader) ->
        preset = loader.dataLoaded ? '# Enter your network definition here.\n# Use Shift+Enter to update the visualization.'
        @editor.setValue(preset)
        #alert(preset)

    onKeyDown: (e) ->
        if (e.shiftKey && e.keyCode==13)
            # Using onKeyDown lets us prevent the default action,
            # even if an error is encountered (say, due to parsing).
            # This would not be possible with keymaps.
            e.preventDefault()
            @loaderFunc @editor.getValue()
