import struct

def pack_isrc(isrc_str):
    """Converts 12-char ISRC string to 64-bit integer (50 bits used)."""
    if not isrc_str or len(isrc_str) != 12:
        return 0
    
    isrc_str = isrc_str.upper()
    try:
        # 1. Country (2 chars A-Z) -> 10 bits
        c1 = ord(isrc_str[0]) - ord('A')
        c2 = ord(isrc_str[1]) - ord('A')
        country = (c1 * 26) + c2
        
        # 2. Registrant (3 chars A-Z, 0-9) -> 16 bits
        def char_to_int(c):
            if 'A' <= c <= 'Z': return ord(c) - ord('A')
            if '0' <= c <= '9': return ord(c) - ord('0') + 26
            return 0
        
        r1 = char_to_int(isrc_str[2])
        r2 = char_to_int(isrc_str[3])
        r3 = char_to_int(isrc_str[4])
        registrant = (r1 * 36 * 36) + (r2 * 36) + r3
        
        # 3. Year (2 digits) -> 7 bits
        year = int(isrc_str[5:7])
        
        # 4. Designation (5 digits) -> 17 bits
        designation = int(isrc_str[7:12])
        
        # Layout: [10 bit country][16 bit registrant][7 bit year][17 bit designation]
        return (country << 40) | (registrant << 24) | (year << 17) | designation
    except:
        return 0

def unpack_isrc(packed):
    """Converts 64-bit integer back to 12-char ISRC string."""
    designation = packed & 0x1FFFF
    year = (packed >> 17) & 0x7F
    registrant = (packed >> 24) & 0xFFFF
    country = (packed >> 40) & 0x3FF
    
    c1 = chr((country // 26) + ord('A'))
    c2 = chr((country % 26) + ord('A'))
    
    def int_to_char(i):
        if i < 26: return chr(i + ord('A'))
        return chr(i - 26 + ord('0'))
    
    r1 = int_to_char(registrant // 1296)
    r2 = int_to_char((registrant // 36) % 36)
    r3 = int_to_char(registrant % 36)
    
    return f"{c1}{c2}{r1}{r2}{r3}{year:02d}{designation:05d}"
