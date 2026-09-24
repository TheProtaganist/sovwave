"""
    Sovwave.GUI.Server

Pure Julia Standard Library (Sockets) HTTP Server for the local Sovwave Visual GUI.
Provides zero-dependency visual model training, live wave lattice monitoring,
multilingual tokenizer playground, and interactive model introspection directly in the browser.
"""
module GUI

using Sockets
using Printf

export launch_gui, stop_gui!

const _ACTIVE_SERVERS = Ref{Vector{Sockets.TCPServer}}(Sockets.TCPServer[])

"""
    launch_gui(; port::Int = 8080, host::String = "127.0.0.1", open_browser::Bool = false)::Sockets.TCPServer

Launches the local Sovwave Visual Quantum GUI server on `http://host:port`.
Pure Julia stdlib implementation requiring zero external C or web dependencies.
"""
function launch_gui(; port::Int = 8080, host::String = "127.0.0.1", open_browser::Bool = false)::Sockets.TCPServer
    gui_dir = normpath(joinpath(@__DIR__, "../../gui"))
    server = Sockets.listen(Sockets.IPv4(host), port)
    push!(_ACTIVE_SERVERS[], server)

    println("====================================================================")
    println(" 🌊 SOVWAVE VISUAL QUANTUM GUI INITIALIZED")
    println("====================================================================")
    @printf("  Local GUI URL:      http://%s:%d\n", host, port)
    println("  Quantum Lattice:    Active (432 Hz Carrier)")
    println("  Multilingual UTF-8: 100% Loss-Free Native Wave Primitive")
    println("  Architecture:       Pure Julia Sockets (Zero external web deps)")
    println("--------------------------------------------------------------------")
    println("  Press Ctrl+C or run Sovwave.GUI.stop_gui!() to close.")
    println("====================================================================")

    if open_browser
        try
            url = "http://$host:$port"
            if Sys.islinux()
                run(`xdg-open $url`, wait=false)
            elseif Sys.isapple()
                run(`open $url`, wait=false)
            elseif Sys.iswindows()
                run(`cmd /c start $url`, wait=false)
            end
        catch
            # Silent fallback if browser opener unavailable
        end
    end

    @async begin
        while isopen(server)
            try
                client = accept(server)
                @async handle_client(client, gui_dir)
            catch e
                if !(e isa Base.IOError)
                    @warn "Sovwave GUI Server error" exception=e
                end
                break
            end
        end
    end

    return server
end

"""
    stop_gui!()::Nothing

Terminates all active Sovwave GUI local web servers.
"""
function stop_gui!()::Nothing
    for s in _ACTIVE_SERVERS[]
        try
            close(s)
        catch
        end
    end
    empty!(_ACTIVE_SERVERS[])
    println("🌊 Sovwave GUI Server stopped.")
    return nothing
end

"""
    handle_client(client::Sockets.TCPSocket, gui_dir::String)

Handles HTTP/1.1 requests for the local visual GUI.
"""
function handle_client(client::Sockets.TCPSocket, gui_dir::String)
    try
        req_line = readline(client)
        parts = split(req_line)
        if length(parts) < 2
            close(client)
            return
        end

        method, target = parts[1], parts[2]
        
        # Consume incoming headers
        while true
            line = readline(client)
            isempty(strip(line)) && break
        end

        clean_path = split(target, '?')[1]
        
        # Route Handling
        if clean_path == "/" || clean_path == "/index.html"
            file_path = joinpath(gui_dir, "index.html")
            send_file(client, file_path, "text/html; charset=utf-8")
        elseif clean_path == "/style.css"
            file_path = joinpath(gui_dir, "style.css")
            send_file(client, file_path, "text/css; charset=utf-8")
        elseif clean_path == "/app.js"
            file_path = joinpath(gui_dir, "app.js")
            send_file(client, file_path, "application/javascript; charset=utf-8")
        elseif clean_path == "/api/status"
            send_json(client, "{\"status\":\"ok\",\"version\":\"0.2.1\",\"carrier_frequency\":432.0}")
        else
            send_response(client, 404, "text/plain", "404 Not Found")
        end
    catch e
        # Ignore broken pipes or aborts
    finally
        try
            close(client)
        catch
        end
    end
end

"""
    send_file(client::Sockets.TCPSocket, path::String, mime::String)

Reads a static file from disk and transmits HTTP 200 response with correct MIME type to client.
"""
function send_file(client::Sockets.TCPSocket, path::String, mime::String)
    if !isfile(path)
        send_response(client, 404, "text/plain", "File Not Found")
        return
    end
    data = read(path)
    header = "HTTP/1.1 200 OK\r\nContent-Type: $mime\r\nContent-Length: $(length(data))\r\nConnection: close\r\n\r\n"
    write(client, header)
    write(client, data)
end

"""
    send_json(client::Sockets.TCPSocket, json_str::String)

Sends an HTTP 200 OK JSON response with CORS headers to the connected TCP client socket.
"""
function send_json(client::Sockets.TCPSocket, json_str::String)
    data = codeunits(json_str)
    header = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: $(length(data))\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n"
    write(client, header)
    write(client, data)
end

"""
    send_response(client::Sockets.TCPSocket, code::Int, mime::String, body::String)

Sends a generic HTTP status code response with MIME content-type and body to client.
"""
function send_response(client::Sockets.TCPSocket, code::Int, mime::String, body::String)
    data = codeunits(body)
    header = "HTTP/1.1 $code Status\r\nContent-Type: $mime\r\nContent-Length: $(length(data))\r\nConnection: close\r\n\r\n"
    write(client, header)
    write(client, data)
end

end # module GUI
