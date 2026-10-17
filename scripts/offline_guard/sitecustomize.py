"""Python socket egress guard. Not a substitute for an OS/network firewall."""
import ipaddress
import os
import socket

if os.getenv('FORTEVISION_OFFLINE') == '1':
    def allowed(host):
        if isinstance(host, bytes):
            host = host.decode('ascii')
        if host in ('localhost', None):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if isinstance(address, tuple) and not allowed(address[0]):
            raise PermissionError('Offline mode: external connections are disabled')
        return original_connect(self, address)

    def connect_ex(self, address):
        if isinstance(address, tuple) and not allowed(address[0]):
            raise PermissionError('Offline mode: external connections are disabled')
        return original_connect_ex(self, address)

    def getaddrinfo(host, *args, **kwargs):
        if host not in ('0.0.0.0', '::') and not allowed(host):
            raise PermissionError('Offline mode: external DNS is disabled')
        return original_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.getaddrinfo = getaddrinfo
    os.environ['FORTEVISION_OFFLINE_GUARD'] = 'active'
